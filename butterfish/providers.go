package butterfish

import (
	"context"
	"errors"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/openai/openai-go"
	openaioption "github.com/openai/openai-go/option"
	"google.golang.org/genai"
)

// ProviderType represents the AI provider type
type ProviderType string

const (
	ProviderOpenAI    ProviderType = "openai"
	ProviderAnthropic ProviderType = "anthropic"
	ProviderGoogle    ProviderType = "google"
)

// ProviderClient wraps underlying SDK clients with provider-specific information
type ProviderClient struct {
	Type       ProviderType
	OpenAI     openai.Client
	Anthropic  anthropic.Client
	Google     *genai.Client
	BaseURL    string // For OpenAI custom base URLs
}

// NewOpenAIProvider creates a new OpenAI provider client
func NewOpenAIProvider(token, baseURL string) (*ProviderClient, error) {
	if token == "" {
		return nil, errors.New("OpenAI token is required")
	}

	opts := []openaioption.RequestOption{openaioption.WithAPIKey(token)}
	if baseURL != "" {
		opts = append(opts, openaioption.WithBaseURL(baseURL))
	}

	client := openai.NewClient(opts...)

	return &ProviderClient{
		Type:    ProviderOpenAI,
		OpenAI:  client,
		BaseURL: baseURL,
	}, nil
}

// NewAnthropicProvider creates a new Anthropic provider client
func NewAnthropicProvider(token string) (*ProviderClient, error) {
	if token == "" {
		return nil, errors.New("Anthropic token is required")
	}

	client := anthropic.NewClient(anthropicoption.WithAPIKey(token))

	return &ProviderClient{
		Type:      ProviderAnthropic,
		Anthropic: client,
	}, nil
}

// NewGoogleProvider creates a new Google provider client
func NewGoogleProvider(token string) (*ProviderClient, error) {
	if token == "" {
		return nil, errors.New("Google token is required")
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  token,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Google client: %w", err)
	}

	return &ProviderClient{
		Type:   ProviderGoogle,
		Google: client,
	}, nil
}

// NewMultiProviderLLM creates a provider client based on configuration
func NewMultiProviderLLM(config *ButterfishConfig) (*ProviderClient, error) {
	provider := ProviderType(config.Provider)
	if provider == "" {
		provider = ProviderOpenAI // default
	}

	switch provider {
	case ProviderOpenAI:
		return NewOpenAIProvider(config.OpenAIToken, config.BaseURL)
	case ProviderAnthropic:
		return NewAnthropicProvider(config.AnthropicToken)
	case ProviderGoogle:
		return NewGoogleProvider(config.GoogleToken)
	default:
		return nil, fmt.Errorf("unknown provider: %s", provider)
	}
}

// ValidateProviderConfig checks if the provider configuration is valid
func ValidateProviderConfig(config *ButterfishConfig) error {
	provider := ProviderType(config.Provider)
	if provider == "" {
		provider = ProviderOpenAI // default
	}

	switch provider {
	case ProviderOpenAI:
		if config.OpenAIToken == "" {
			return errors.New("OpenAI token is required when using OpenAI provider")
		}
	case ProviderAnthropic:
		if config.AnthropicToken == "" {
			return errors.New("Anthropic token is required when using Anthropic provider")
		}
	case ProviderGoogle:
		if config.GoogleToken == "" {
			return errors.New("Google token is required when using Google provider")
		}
	default:
		return fmt.Errorf("unknown provider: %s", provider)
	}

	return nil
}

// GetProviderName returns the provider name as a string
func (p *ProviderClient) GetProviderName() string {
	return string(p.Type)
}

