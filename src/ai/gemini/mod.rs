#[cfg(test)]
macro_rules! impl_with_gemini_base_url {
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
pub(crate) use impl_with_gemini_base_url;

#[cfg(test)]
pub(crate) mod test_support {
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockBuilder};

    pub(crate) const GENERATE_CONTENT_PATH_REGEX: &str = r"/v1beta/models/.+:generateContent";
    pub(crate) const BATCH_EMBED_CONTENTS_PATH_REGEX: &str =
        r"/v1beta/models/.+:batchEmbedContents";

    pub(crate) fn post_path_regex(path_pattern: &'static str) -> MockBuilder {
        Mock::given(method("POST")).and(path_regex(path_pattern))
    }
}

pub mod chat;
pub mod client;
pub mod embedding;
pub mod image;
pub mod qa;
pub mod types;

pub use chat::GeminiChatClient;
pub use embedding::{GeminiEmbeddingClient, GEMINI_MAX_BATCH_EMBED_ITEMS};
pub use image::GeminiImageClient;
pub use qa::GeminiImageQaClient;
