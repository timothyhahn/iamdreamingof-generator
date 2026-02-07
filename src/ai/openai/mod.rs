#[cfg(test)]
macro_rules! impl_with_openai_base_url {
    ($client:ty) => {
        impl $client {
            fn with_base_url(mut self, base_url: String) -> Self {
                self.http = self.http.with_base_url(base_url);
                self
            }
        }
    };
}

#[cfg(test)]
pub(crate) use impl_with_openai_base_url;

#[cfg(test)]
pub(crate) mod test_support {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockBuilder};

    pub(crate) const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";
    pub(crate) const IMAGES_GENERATIONS_PATH: &str = "/v1/images/generations";
    pub(crate) const EMBEDDINGS_PATH: &str = "/v1/embeddings";

    pub(crate) fn post(api_path: &'static str) -> MockBuilder {
        Mock::given(method("POST")).and(path(api_path))
    }
}

pub mod chat;
pub mod client;
pub mod embedding;
pub mod image;
pub mod qa;
pub mod types;

pub use chat::OpenAiChatClient;
pub use embedding::OpenAiEmbeddingClient;
pub use image::OpenAiImageClient;
pub use qa::OpenAiImageQaClient;
