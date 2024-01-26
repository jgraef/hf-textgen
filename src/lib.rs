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
//! 
//! let token_stream = model.generate("[INST] Write a short poem about AI. [/INST]").await?;
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
    Stream,
    StreamExt,
};
use reqwest::{
    header::{
        self,
        HeaderValue,
    },
    Client,
};
use serde::{
    Deserialize,
    Serialize,
};

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

/// A client for the HuggingFace text generation API.
#[derive(Debug, Clone)]
pub struct Api {
    client: Client,
}

impl Api {
    const BASE_URL: &'static str = "https://api-inference.huggingface.co/";

    /// Creates a new API instance.
    /// 
    /// # Arguments
    /// 
    /// - `hf_token`: (optional) Your HuggingFace API token. (starts with `hf_`).
    pub fn new(hf_token: Option<String>) -> Self {
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

        Self { client }
    }

    /// Creates a [`TextGeneration`] object for the given `model_id`.
    pub fn text_generation(&self, model_id: &str) -> TextGeneration {
        let url = format!("{}models/{}", Api::BASE_URL, model_id);
        TextGeneration::new(self.client.clone(), url)
    }
}

impl Default for Api {
    fn default() -> Self {
        Self::new(None)
    }
}

/// A model endpoint for text generation.
#[derive(Clone, Debug)]
pub struct TextGeneration {
    client: Client,
    url: String,

    /// Integer to define the top tokens considered within the sample operation to create new text.
    pub top_k: Option<usize>,

    /// Float to define the tokens that are within the sample operation of text generation. Add tokens in the sample for more probable to least probable until the sum of the probabilities is greater than top_p.
    pub top_p: Option<f32>,

    /// (Default: 1.0). Float (0.0-100.0). The temperature of the sampling
    /// operation. 1 means regular sampling, 0 means always take the highest
    /// score, 100.0 is getting closer to uniform probability.
    pub temparature: f32,

    /// (Default: 250). The max number of tokens to generate. Although the HuggingFace API docs
    /// state that this is limited to a max of 250 tokens, you might be able to
    /// use higher values.
    pub max_new_tokens: Option<usize>,

    /// Float (0.0-100.0). The more a token is used within generation the more it is penalized to not be picked in successive generation passes.
    pub repetition_penalty: Option<f32>,

    /// (Default: true) Boolean. If the model is not ready, wait for it instead of receiving 503. It limits the number of requests required to get your inference done. It is advised to only set this flag to true after receiving a 503 error as it will limit hanging in your application to known places.
    pub use_cache: bool,
}

impl TextGeneration {
    fn new(client: Client, url: String) -> Self {
        Self {
            client,
            url,
            top_k: None,
            top_p: None,
            temparature: 1.0,
            max_new_tokens: None,
            repetition_penalty: None,
            use_cache: true,
        }
    }

    /// Calls the HuggingFace text generation API to generate text given a
    /// `prompt`. The `prompt` itself will not be included in the output.
    ///
    /// This returns a [`TokenStream`], which yields [`Token`]s with the
    /// associated token ID and text. If you only need the text (and ignoring
    /// special tokens), you can use [`TokenStream::text`] to get a stream that
    /// yields strings.
    pub async fn generate(&self, prompt: &str) -> Result<TokenStream, Error> {
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
            stream: true,
        };

        let stream = self
            .client
            .post(&self.url)
            .json(&request)
            .send()
            .await?
            .bytes_stream()
            .eventsource();

        Ok(TokenStream {
            inner: Box::pin(stream),
        })
    }
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
}
