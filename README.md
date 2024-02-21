An API client for the free [HuggingFace text generation API](https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task).

# Example

```rust
use std::{io::{stdout, Write}, error::Error};
use futures::stream::TryStreamExt;
use hf_textgen::Api;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let api = Api::default();
    let model = api.text_generation("mistralai/Mistral-7B-Instruct-v0.2");

    let token_stream = model.generate_stream("[INST] Write a short poem about AI. [/INST]").await?;
    let mut text_stream = token_stream.text();

    while let Some(text) = text_stream.try_next().await? {
        print!("{text}");
        stdout().flush()?;
    }

    Ok(())
}
```
