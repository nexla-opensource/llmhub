class LLMHubError(RuntimeError):
    pass

class ProviderNotAvailable(LLMHubError):
    pass

class FeatureNotSupported(LLMHubError):
    pass

class StructuredOutputValidationError(LLMHubError):
    pass
