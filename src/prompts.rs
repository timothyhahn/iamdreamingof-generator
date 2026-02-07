pub const CHAT_SYSTEM: &str = include_str!("../data/prompts/chat_system.txt");
pub const CHAT_USER: &str = include_str!("../data/prompts/chat_user.txt");
pub const IMAGE_ENHANCEMENT: &str = include_str!("../data/prompts/image_enhancement.txt");
pub const QA_SYSTEM: &str = include_str!("../data/prompts/qa_system.txt");
pub const QA_USER: &str = include_str!("../data/prompts/qa_user.txt");

/// Replace `{{key}}` placeholders in a template string.
pub fn render(template: &str, vars: &[(&str, &str)]) -> String {
    let mut result = template.to_string();
    for (key, value) in vars {
        result = result.replace(&format!("{{{{{}}}}}", key), value);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_single_var() {
        assert_eq!(
            render("Hello {{name}}!", &[("name", "world")]),
            "Hello world!"
        );
    }

    #[test]
    fn test_render_multiple_vars() {
        assert_eq!(
            render("{{a}} and {{b}}", &[("a", "cats"), ("b", "dogs")]),
            "cats and dogs"
        );
    }

    #[test]
    fn test_prompts_are_non_empty() {
        assert!(!CHAT_SYSTEM.is_empty());
        assert!(!CHAT_USER.is_empty());
        assert!(!IMAGE_ENHANCEMENT.is_empty());
        assert!(!QA_SYSTEM.is_empty());
        assert!(!QA_USER.is_empty());
    }

    #[test]
    fn test_chat_user_has_words_placeholder() {
        assert!(CHAT_USER.contains("{{words}}"));
    }

    #[test]
    fn test_image_enhancement_has_placeholders() {
        assert!(IMAGE_ENHANCEMENT.contains("{{prompt}}"));
        assert!(IMAGE_ENHANCEMENT.contains("{{words}}"));
    }
}
