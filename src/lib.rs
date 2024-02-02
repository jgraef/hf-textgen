//! An API client for the free [HuggingFace text generation API](https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task).
//!
//! # Example
//!
//! ```
//! # use std::{io::{stdout, Write}, error::Error};
//! # use futures::stream::TryStreamExt;
//! # use hf_textgen::Api;
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), Box<dyn Error>> {
//! let api = Api::default();
//! let model = api.text_generation("mistralai/Mistral-7B-Instruct-v0.2");
//! # let mut model = model;
//! # model.max_new_tokens = Some(10);
//!
//! let token_stream = model
//!     .generate("[INST] Write a short poem about AI. [/INST]")
//!     .await?;
//! let mut text_stream = token_stream.text();
//!
//! while let Some(text) = text_stream.try_next().await? {
//!     print!("{text}");
//!     stdout().flush()?;
//! }
//! # Ok(())
//! # }
//! ```

use std::{
    pin::Pin,
    sync::Arc,
    task::{
        Context,
        Poll,
    },
};

use eventsource_stream::{
    Event,
    EventStreamError,
    Eventsource,
};
use futures::{
    Future,
    FutureExt,
    Stream,
    StreamExt,
};
use reqwest::{
    header::{
        self,
        HeaderMap,
        HeaderValue,
        LINK,
    },
    Client,
    Response,
};
use serde::{
    Deserialize,
    Serialize,
};
use url::Url;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("http error")]
    Reqwest(#[from] reqwest::Error),

    #[error("event stream error")]
    EventStream(#[from] EventStreamError<reqwest::Error>),

    #[error("json error")]
    Json(#[from] serde_json::Error),

    #[error("api returned invalid response")]
    InvalidResponse,
}

#[derive(Debug)]
struct ApiInner {
    client: Client,
    base_url: String,
}

/// A client for the HuggingFace text generation API.
#[derive(Debug, Clone)]
pub struct Api {
    inner: Arc<ApiInner>,
}

impl Api {
    pub const DEFAULT_BASE_URL: &'static str = "https://api-inference.huggingface.co";

    /// Creates a new API instance.
    ///
    /// # Arguments
    ///
    /// - `hf_token`: (optional) Your HuggingFace API token. (starts with
    ///   `hf_`).
    fn new(base_url: Option<String>, hf_token: Option<String>) -> Self {
        let mut builder = Client::builder();

        let token = if let Some(token) = hf_token {
            Some(token)
        }
        else {
            std::env::var("HF_TOKEN").ok()
        };

        if let Some(token) = token {
            let mut headers = header::HeaderMap::new();
            headers.insert(
                "Authorization",
                HeaderValue::from_str(format!("Bearer {}", token).as_str())
                    .expect("failed to create Authorization header"),
            );
            builder = builder.default_headers(headers);
        }

        let client = builder.build().expect("http client builder failed");

        let base_url = base_url.unwrap_or_else(|| Self::DEFAULT_BASE_URL.to_owned());

        Self {
            inner: Arc::new(ApiInner { client, base_url }),
        }
    }

    pub fn builder() -> ApiBuilder {
        ApiBuilder::default()
    }

    /// Creates a [`TextGeneration`] object for the given `model_id`.
    pub fn text_generation(&self, model_id: &str) -> TextGeneration {
        TextGeneration::new(self.clone(), model_id.to_owned())
    }

    pub fn list_models(
        &self,
        search: Option<&str>,
        author: Option<&str>,
        tags: &[&str],
        limit: Option<usize>,
    ) -> ModelList {
        ModelList::new(self.clone(), search, author, tags, limit)
    }

    pub async fn quick_search(&self, search: &str) -> Result<QuickSearchResult, Error> {
        #[derive(Debug, Serialize)]
        struct Query<'a> {
            q: &'a str,
            r#type: &'a str,
        }

        let result: QuickSearchResult = self
            .inner
            .client
            .get("https://huggingface.co/api/quicksearch")
            .query(&Query {
                q: search,
                r#type: "model",
            })
            .send()
            .await?
            .json()
            .await?;

        Ok(QuickSearchResult {
            models: result.models,
            models_count: result.models_count,
        })
    }
}

impl Default for Api {
    fn default() -> Self {
        Self::new(None, None)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ApiBuilder {
    base_url: Option<String>,
    hf_token: Option<String>,
}

impl ApiBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = Some(base_url);
        self
    }

    pub fn with_hf_token(mut self, hf_token: String) -> Self {
        self.hf_token = Some(hf_token);
        self
    }

    pub fn build(self) -> Api {
        Api::new(self.base_url, self.hf_token)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub likes: usize,
    #[serde(default)]
    pub downloads: usize,
    #[serde(default)]
    pub tags: Vec<String>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub created_at: String,
}

pub struct ModelList {
    api: Api,
    base_url: Url,
    next_url: Option<Url>,
    send_future: Option<Pin<Box<dyn Future<Output = Result<Response, reqwest::Error>>>>>,
    response_future: Option<Pin<Box<dyn Future<Output = Result<Vec<ModelInfo>, reqwest::Error>>>>>,
    buffer: Vec<ModelInfo>,
}

impl ModelList {
    pub fn new(
        api: Api,
        search: Option<&str>,
        author: Option<&str>,
        tags: &[&str],
        limit: Option<usize>,
    ) -> Self {
        let tags = if tags.is_empty() {
            None
        }
        else {
            Some(tags.join(","))
        };

        #[derive(Debug, Serialize)]
        struct Query<'a> {
            search: Option<&'a str>,
            author: Option<&'a str>,
            tags: Option<String>,
            limit: Option<usize>,
        }

        let url = "https://huggingface.co/api/models";
        let send_future = api
            .inner
            .client
            .get(url)
            .query(&Query {
                search,
                author,
                tags,
                limit,
            })
            .send();

        let base_url = Url::parse(url).unwrap();

        Self {
            api,
            base_url,
            next_url: None,
            send_future: Some(Box::pin(send_future)),
            response_future: None,
            buffer: vec![],
        }
    }
}

impl Stream for ModelList {
    type Item = Result<ModelInfo, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        fn parse_link_header(headers: &HeaderMap, base_url: &Url) -> Option<Url> {
            let link_header = headers.get(LINK)?.to_str().ok()?;
            let link_header = http_link::parse_link_header(link_header, base_url).ok()?;
            let link = link_header
                .into_iter()
                .find_map(|link| (link.rel == "next").then(move || link.target))?;
            Some(link)
        }

        loop {
            // first return everything we have buffered
            if let Some(item) = self.buffer.pop() {
                return Poll::Ready(Some(Ok(item)));
            }

            // send the request
            if let Some(send_future) = &mut self.send_future {
                match send_future.poll_unpin(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Err(error)) => {
                        self.send_future = None;
                        return Poll::Ready(Some(Err(error.into())));
                    }
                    Poll::Ready(Ok(response)) => {
                        self.send_future = None;
                        self.next_url = parse_link_header(response.headers(), &self.base_url);
                        self.response_future = Some(Box::pin(response.json()));
                    }
                }
            }

            // wait for response and parse it
            if let Some(response_future) = &mut self.response_future {
                match response_future.poll_unpin(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Err(error)) => {
                        self.response_future = None;
                        return Poll::Ready(Some(Err(error.into())));
                    }
                    Poll::Ready(Ok(response)) => {
                        self.response_future = None;
                        assert!(self.buffer.is_empty());
                        self.buffer = response;
                        self.buffer.reverse();

                        // we continue here, to pull the first item from the buffer
                        continue;
                    }
                }
            }

            // start the next request
            if let Some(next_url) = self.next_url.take() {
                self.send_future = Some(Box::pin(self.api.inner.client.get(next_url).send()));
                continue;
            }

            // if we end up here, the buffer is empty, no future is set, and we don't have a
            // next url
            return Poll::Ready(None);
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuickSearchResult {
    pub models: Vec<ModelInfo>,
    pub models_count: usize,
}

/// A model endpoint for text generation.
#[derive(Clone, Debug)]
pub struct TextGeneration {
    api: Api,

    model_id: String,

    /// Integer to define the top tokens considered within the sample operation
    /// to create new text.
    pub top_k: Option<usize>,

    /// Float to define the tokens that are within the sample operation of text
    /// generation. Add tokens in the sample for more probable to least probable
    /// until the sum of the probabilities is greater than top_p.
    pub top_p: Option<f32>,

    /// (Default: 1.0). Float (0.0-100.0). The temperature of the sampling
    /// operation. 1 means regular sampling, 0 means always take the highest
    /// score, 100.0 is getting closer to uniform probability.
    pub temparature: f32,

    /// (Default: 250). The max number of tokens to generate. Although the
    /// HuggingFace API docs state that this is limited to a max of 250
    /// tokens, you might be able to use higher values.
    pub max_new_tokens: Option<usize>,

    /// Float (0.0-100.0). The more a token is used within generation the more
    /// it is penalized to not be picked in successive generation passes.
    pub repetition_penalty: Option<f32>,

    /// (Default: true) Boolean. If the model is not ready, wait for it instead
    /// of receiving 503. It limits the number of requests required to get your
    /// inference done. It is advised to only set this flag to true after
    /// receiving a 503 error as it will limit hanging in your application to
    /// known places.
    pub use_cache: bool,
}

impl TextGeneration {
    fn new(api: Api, model_id: String) -> Self {
        Self {
            api,
            model_id,
            top_k: None,
            top_p: None,
            temparature: 1.0,
            max_new_tokens: None,
            repetition_penalty: None,
            use_cache: true,
        }
    }

    async fn send_request(&self, prompt: &str, stream: bool) -> Result<Response, Error> {
        #[derive(Serialize)]
        struct Parameters {
            return_full_text: bool,
            max_new_tokens: usize,
            temperature: f32,
            #[serde(skip_serializing_if = "Option::is_none")]
            top_k: Option<usize>,
            #[serde(skip_serializing_if = "Option::is_none")]
            top_p: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            repetition_penalty: Option<f32>,
        }

        #[derive(Serialize)]
        struct Options {
            wait_for_model: bool,
            use_cache: bool,
        }

        #[derive(Serialize)]
        struct Request<'a> {
            inputs: &'a str,
            parameters: Parameters,
            options: Options,
            stream: bool,
        }

        let request = Request {
            inputs: prompt,
            parameters: Parameters {
                return_full_text: false,
                max_new_tokens: self.max_new_tokens.unwrap_or(250),
                temperature: self.temparature,
                top_k: self.top_k,
                top_p: self.top_p,
                repetition_penalty: self.repetition_penalty,
            },
            options: Options {
                wait_for_model: true,
                use_cache: self.use_cache,
            },
            stream,
        };

        let url = format!("{}/models/{}", self.api.inner.base_url, self.model_id);

        let response = self
            .api
            .inner
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        Ok(response)
    }

    /// Calls the HuggingFace text generation API to generate text given a
    /// `prompt`. The `prompt` itself will not be included in the output.
    ///
    /// This returns a [`TokenStream`], which yields [`Token`]s with the
    /// associated token ID and text. If you only need the text (and ignoring
    /// special tokens), you can use [`TokenStream::text`] to get a stream that
    /// yields strings.
    pub async fn generate(&self, prompt: &str) -> Result<TokenStream, Error> {
        let stream = self
            .send_request(prompt, true)
            .await?
            .bytes_stream()
            .eventsource();

        Ok(TokenStream {
            inner: Box::pin(stream),
        })
    }

    pub async fn status(&self) -> Result<ModelStatus, Error> {
        let url = format!("{}/status/{}", self.api.inner.base_url, self.model_id);
        let status: ModelStatus = self.api.inner.client.get(&url).send().await?.json().await?;
        Ok(status)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelStatus {
    pub loaded: bool,
    pub state: ModelState,
    pub compute_type: ComputeType,
    pub framework: String,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelState {
    TooBig,
    Loadable,
    // todo: what other states are there?
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ComputeType {
    Cpu,
    Gpu,
}

/// A stream of tokens returned by the API.
pub struct TokenStream {
    inner: Pin<Box<dyn Stream<Item = Result<Event, EventStreamError<reqwest::Error>>> + 'static>>,
}

impl TokenStream {
    pub fn text(self) -> TextStream {
        TextStream { inner: self }
    }
}

impl Stream for TokenStream {
    type Item = Result<Token, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match self.inner.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(event))) => {
                    match serde_json::from_str::<TextGenerationEvent>(&event.data) {
                        Ok(TextGenerationEvent {
                            token: Some(token), ..
                        }) => {
                            break Poll::Ready(Some(Ok(token)));
                        }
                        Ok(_) => {
                            // no token in event, so we poll for the next event.
                        }
                        Err(e) => break Poll::Ready(Some(Err(e.into()))),
                    }
                }
                Poll::Ready(Some(Err(e))) => break Poll::Ready(Some(Err(e.into()))),
                Poll::Ready(None) => break Poll::Ready(None),
                Poll::Pending => break Poll::Pending,
            }
        }
    }
}

/// A text stream returned by the API.
pub struct TextStream {
    inner: TokenStream,
}

impl TextStream {
    pub fn into_inner(self) -> TokenStream {
        self.inner
    }
}

impl Stream for TextStream {
    type Item = Result<String, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match self.inner.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(token))) => {
                    if !token.special {
                        break Poll::Ready(Some(Ok(token.text)));
                    }
                }
                Poll::Ready(Some(Err(e))) => break Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => break Poll::Ready(None),
                Poll::Pending => break Poll::Pending,
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct TextGenerationEvent {
    token: Option<Token>,
}

/// A single token returned by the API
#[derive(Debug, Serialize, Deserialize)]
pub struct Token {
    /// The model-internal ID for this token.
    pub id: u32,

    /// Log-probability for this token.
    pub logprob: f32,

    /// Textual representation of this token.
    pub text: String,

    /// Whether this token is a special token.
    pub special: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn it_generates_a_poem() {
        let api = Api::default();
        let mut model = api.text_generation("mistralai/Mistral-7B-Instruct-v0.2");
        model.max_new_tokens = Some(10);
        let stream = model
            .generate("[INST] Write a short poem about AI. [/INST]")
            .await
            .unwrap();
        let output = stream
            .text()
            .map(|text| text.unwrap())
            .collect::<String>()
            .await;
        assert_eq!(output, " In silicon valleys deep, where thoughts");
    }

    #[tokio::test]
    async fn it_replies_with_a_status() {
        let api = Api::default();
        let model = api.text_generation("mistralai/Mistral-7B-Instruct-v0.2");
        let _status = model.status().await.unwrap();
    }
}
