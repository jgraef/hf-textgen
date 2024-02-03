use std::error::Error;

use futures::stream::TryStreamExt;
use hf_textgen::Api;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let api = Api::default();
    let mut models = api.list_models(
        Some("Mistral"),
        None,
        &[
            "text-generation",
            "text-generation-inference",
            "endpoints_compatible",
        ],
        Some(10),
    );

    while let Some(model_info) = models.try_next().await? {
        println!("{}", model_info.id);
    }

    Ok(())
}
