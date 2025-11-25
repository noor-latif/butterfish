package butterfish

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicparam "github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/bakks/butterfish/util"
	"github.com/coder/aisdk-go"
	"github.com/openai/openai-go"
	openaiparam "github.com/openai/openai-go/packages/param"
	"google.golang.org/genai"
)

// AISDKLLM implements the LLM interface using aisdk-go for multi-provider support
type AISDKLLM struct {
	provider *ProviderClient
	verbose  bool
}

// NewAISDKLLM creates a new AISDKLLM instance
func NewAISDKLLM(provider *ProviderClient, verbose bool) *AISDKLLM {
	return &AISDKLLM{
		provider: provider,
		verbose:  verbose,
	}
}

// isLegacyCompletionModel checks if the model requires the legacy completions API
func isLegacyCompletionModel(model string) bool {
	return strings.HasSuffix(model, "-instruct") || model == "text-davinci-003" || model == "text-davinci-002"
}

// Completion implements the LLM interface for non-streaming completions
func (a *AISDKLLM) Completion(request *util.CompletionRequest) (*util.CompletionResponse, error) {
	// Handle legacy completion models (like gpt-3.5-turbo-instruct) separately
	if a.provider.Type == ProviderOpenAI && isLegacyCompletionModel(request.Model) {
		return a.legacyCompletionOpenAI(request.Ctx, request)
	}

	messages, err := ConvertCompletionRequestToAISDKMessages(request)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages: %w", err)
	}

	switch a.provider.Type {
	case ProviderOpenAI:
		return a.completionOpenAI(request.Ctx, request, messages)
	case ProviderAnthropic:
		return a.completionAnthropic(request.Ctx, request, messages)
	case ProviderGoogle:
		return a.completionGoogle(request.Ctx, request, messages)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", a.provider.Type)
	}
}

// CompletionStream implements the LLM interface for streaming completions
func (a *AISDKLLM) CompletionStream(request *util.CompletionRequest, writer io.Writer) (*util.CompletionResponse, error) {
	// Handle legacy completion models (like gpt-3.5-turbo-instruct) separately
	if a.provider.Type == ProviderOpenAI && isLegacyCompletionModel(request.Model) {
		return a.legacyCompletionStreamOpenAI(request.Ctx, request, writer)
	}

	messages, err := ConvertCompletionRequestToAISDKMessages(request)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages: %w", err)
	}

	switch a.provider.Type {
	case ProviderOpenAI:
		return a.completionStreamOpenAI(request.Ctx, request, messages, writer)
	case ProviderAnthropic:
		return a.completionStreamAnthropic(request.Ctx, request, messages, writer)
	case ProviderGoogle:
		return a.completionStreamGoogle(request.Ctx, request, messages, writer)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", a.provider.Type)
	}
}

// Embeddings implements the LLM interface for embeddings
func (a *AISDKLLM) Embeddings(ctx context.Context, input []string, verbose bool) ([][]float32, error) {
	switch a.provider.Type {
	case ProviderOpenAI:
		return a.embeddingsOpenAI(ctx, input, verbose)
	case ProviderAnthropic:
		return nil, errors.New("Anthropic does not currently support embeddings API")
	case ProviderGoogle:
		return nil, errors.New("Google does not currently support embeddings API")
	default:
		return nil, fmt.Errorf("unsupported provider: %s", a.provider.Type)
	}
}

// Legacy OpenAI completion (for instruct models like gpt-3.5-turbo-instruct)
func (a *AISDKLLM) legacyCompletionOpenAI(ctx context.Context, request *util.CompletionRequest) (*util.CompletionResponse, error) {
	params := openai.CompletionNewParams{
		Model:     openai.CompletionNewParamsModel(request.Model),
		Prompt:    openai.CompletionNewParamsPromptUnion{OfString: openaiparam.NewOpt(request.Prompt)},
		MaxTokens: openaiparam.NewOpt(int64(request.MaxTokens)),
	}
	if request.Temperature > 0 {
		params.Temperature = openaiparam.NewOpt(float64(request.Temperature))
	}

	if a.verbose {
		log.Printf("Legacy OpenAI Completion Request: model=%s, max_tokens=%d, temperature=%v",
			request.Model, request.MaxTokens, request.Temperature)
	}

	var resp *openai.Completion
	err := withExponentialBackoff(func() error {
		var innerErr error
		resp, innerErr = a.provider.OpenAI.Completions.New(ctx, params)
		return innerErr
	})
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, errors.New("no completions returned from OpenAI")
	}

	responseText := strings.TrimSpace(resp.Choices[0].Text)

	response := &util.CompletionResponse{
		Completion: responseText,
	}

	if a.verbose {
		LogCompletionResponse(*response, resp.ID)
	}

	return response, nil
}

// OpenAI completion (non-streaming)
func (a *AISDKLLM) completionOpenAI(ctx context.Context, request *util.CompletionRequest, messages []aisdk.Message) (*util.CompletionResponse, error) {
	openaiMessages, err := aisdk.MessagesToOpenAI(messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to OpenAI format: %w", err)
	}

	// Convert tools if provided
	var tools []openai.ChatCompletionToolParam
	if len(request.Tools) > 0 {
		aisdkTools := convertToAISDKTools(request.Tools)
		tools = aisdk.ToolsToOpenAI(aisdkTools)
	}

	params := openai.ChatCompletionNewParams{
		Model:     openai.ChatModel(request.Model),
		Messages:  openaiMessages,
		MaxTokens: openaiparam.NewOpt(int64(request.MaxTokens)),
	}
	if request.Temperature > 0 {
		params.Temperature = openaiparam.NewOpt(float64(request.Temperature))
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	if a.verbose {
		LogChatCompletionRequestOpenAI(params)
	}

	var resp *openai.ChatCompletion
	err = withExponentialBackoff(func() error {
		var innerErr error
		resp, innerErr = a.provider.OpenAI.Chat.Completions.New(ctx, params)
		return innerErr
	})
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, errors.New("no completions returned from OpenAI")
	}

	choice := resp.Choices[0]
	responseText := choice.Message.Content // Content is a string in the response

	response := &util.CompletionResponse{
		Completion: responseText,
	}

	// Handle function/tool calls
	if len(choice.Message.ToolCalls) > 0 {
		// For now, handle first tool call (legacy function calling)
		toolCall := choice.Message.ToolCalls[0]
		response.FunctionName = toolCall.Function.Name
		response.FunctionParameters = toolCall.Function.Arguments

		// Convert to ToolCalls format
		response.ToolCalls = []*util.ToolCall{}
		for _, tc := range choice.Message.ToolCalls {
			var args map[string]any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				args = make(map[string]any)
			}
			response.ToolCalls = append(response.ToolCalls, &util.ToolCall{
				Id:   tc.ID,
				Type: "function",
				Function: util.FunctionCall{
					Name:       tc.Function.Name,
					Parameters: tc.Function.Arguments,
				},
			})
		}
	}

	if a.verbose {
		LogCompletionResponse(*response, resp.ID)
	}

	return response, nil
}

// Legacy OpenAI streaming completion (for instruct models like gpt-3.5-turbo-instruct)
func (a *AISDKLLM) legacyCompletionStreamOpenAI(ctx context.Context, request *util.CompletionRequest, writer io.Writer) (*util.CompletionResponse, error) {
	params := openai.CompletionNewParams{
		Model:     openai.CompletionNewParamsModel(request.Model),
		Prompt:    openai.CompletionNewParamsPromptUnion{OfString: openaiparam.NewOpt(request.Prompt)},
		MaxTokens: openaiparam.NewOpt(int64(request.MaxTokens)),
	}
	if request.Temperature > 0 {
		params.Temperature = openaiparam.NewOpt(float64(request.Temperature))
	}

	if a.verbose {
		log.Printf("Legacy OpenAI Streaming Completion Request: model=%s, max_tokens=%d, temperature=%v",
			request.Model, request.MaxTokens, request.Temperature)
	}

	stream := a.provider.OpenAI.Completions.NewStreaming(ctx, params)
	var responseContent strings.Builder

	// Iterate over stream
	for stream.Next() {
		completion := stream.Current()
		if len(completion.Choices) > 0 {
			text := completion.Choices[0].Text
			writer.Write([]byte(text))
			responseContent.WriteString(text)
		}
	}

	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("stream error: %w", err)
	}

	fmt.Fprintf(writer, "\n")

	responseText := strings.TrimSpace(responseContent.String())
	response := &util.CompletionResponse{
		Completion: responseText,
	}

	if a.verbose {
		LogCompletionResponse(*response, "stream-completion")
	}

	return response, nil
}

// OpenAI streaming completion
func (a *AISDKLLM) completionStreamOpenAI(ctx context.Context, request *util.CompletionRequest, messages []aisdk.Message, writer io.Writer) (*util.CompletionResponse, error) {
	openaiMessages, err := aisdk.MessagesToOpenAI(messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to OpenAI format: %w", err)
	}

	// Convert tools if provided
	var tools []openai.ChatCompletionToolParam
	if len(request.Tools) > 0 {
		aisdkTools := convertToAISDKTools(request.Tools)
		tools = aisdk.ToolsToOpenAI(aisdkTools)
	}

	params := openai.ChatCompletionNewParams{
		Model:     openai.ChatModel(request.Model),
		Messages:  openaiMessages,
		MaxTokens: openaiparam.NewOpt(int64(request.MaxTokens)),
	}
	if request.Temperature > 0 {
		params.Temperature = openaiparam.NewOpt(float64(request.Temperature))
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	if a.verbose {
		LogChatCompletionRequestOpenAI(params)
	}

	// Create streaming request
	stream := a.provider.OpenAI.Chat.Completions.NewStreaming(ctx, params)

	// Convert to aisdk DataStream
	dataStream := aisdk.OpenAIToDataStream(stream)

	// Accumulate response
	var responseContent strings.Builder
	var functionName string
	var functionArgs strings.Builder
	var toolCalls []*util.ToolCall
	var id string

	// Handle token timeout
	innerCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	gotChunk := make(chan bool)
	defer close(gotChunk)
	var chunkTimeoutErr error

	timeoutRoutine := func() {
		if request.TokenTimeout == 0 {
			return
		}
		select {
		case <-time.After(request.TokenTimeout):
			chunkTimeoutErr = fmt.Errorf("timed out waiting for streaming response, timeout set to %v", request.TokenTimeout)
			cancel()
		case <-innerCtx.Done():
		case <-gotChunk:
		}
	}

	if request.TokenTimeout > 0 {
		go timeoutRoutine()
	}

	// Process stream
	for part, err := range dataStream {
		if err != nil {
			if chunkTimeoutErr != nil {
				return nil, chunkTimeoutErr
			}
			// Log the error for debugging
			if a.verbose {
				log.Printf("Stream error: %v", err)
			}
			return nil, fmt.Errorf("stream error: %w", err)
		}

		if request.TokenTimeout > 0 {
			select {
			case gotChunk <- true:
			default:
			}
			go timeoutRoutine()
		}

		switch p := part.(type) {
		case aisdk.TextStreamPart:
			writer.Write([]byte(p.Content))
			responseContent.WriteString(p.Content)

		case aisdk.ToolCallStartStreamPart:
			// Initialize new tool call
			for len(toolCalls) <= 0 {
				toolCalls = append(toolCalls, &util.ToolCall{})
			}
			if len(toolCalls) > 0 {
				toolCalls[0].Id = p.ToolCallID
				toolCalls[0].Function.Name = p.ToolName
			}
			writer.Write([]byte(p.ToolName))
			writer.Write([]byte("("))

		case aisdk.ToolCallDeltaStreamPart:
			functionArgs.WriteString(p.ArgsTextDelta)
			writer.Write([]byte(p.ArgsTextDelta))

		case aisdk.ToolCallStreamPart:
			if len(toolCalls) == 0 {
				toolCalls = append(toolCalls, &util.ToolCall{})
			}
			toolCalls[0].Id = p.ToolCallID
			toolCalls[0].Function.Name = p.ToolName
			argsJSON, _ := json.Marshal(p.Args)
			toolCalls[0].Function.Parameters = string(argsJSON)
			functionName = p.ToolName
			functionArgs.Reset()
			functionArgs.WriteString(string(argsJSON))

		case aisdk.FinishStepStreamPart, aisdk.FinishMessageStreamPart:
			// Stream finished
			if _, ok := part.(aisdk.FinishMessageStreamPart); ok {
				id = "stream-completion"
				if a.verbose {
					LogCompletionResponse(util.CompletionResponse{
						Completion: responseContent.String(),
					}, id)
				}
			}
		}
	}

	if functionName != "" || len(toolCalls) > 0 {
		writer.Write([]byte(")"))
	}
	fmt.Fprintf(writer, "\n")

	response := &util.CompletionResponse{
		Completion:         responseContent.String(),
		FunctionName:       functionName,
		FunctionParameters: functionArgs.String(),
		ToolCalls:          toolCalls,
	}

	return response, nil
}

// Anthropic completion (non-streaming)
func (a *AISDKLLM) completionAnthropic(ctx context.Context, request *util.CompletionRequest, messages []aisdk.Message) (*util.CompletionResponse, error) {
	anthropicMessages, system, err := aisdk.MessagesToAnthropic(messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to Anthropic format: %w", err)
	}

	// Convert tools if provided
	var tools []anthropic.ToolUnionParam
	if len(request.Tools) > 0 {
		aisdkTools := convertToAISDKTools(request.Tools)
		tools = aisdk.ToolsToAnthropic(aisdkTools)
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(request.Model),
		Messages: anthropicMessages,
		System:   system,
		MaxTokens: int64(request.MaxTokens),
	}
	if len(tools) > 0 {
		params.Tools = tools
	}
	if request.Temperature > 0 {
		params.Temperature = anthropicparam.NewOpt(float64(request.Temperature))
	}

	if a.verbose {
		LogChatCompletionRequestAnthropic(params)
	}

	var resp *anthropic.Message
	err = withExponentialBackoff(func() error {
		var innerErr error
		resp, innerErr = a.provider.Anthropic.Messages.New(ctx, params)
		return innerErr
	})
	if err != nil {
		return nil, err
	}

	// Extract text content from Anthropic response
	responseText := ""
	toolCalls := []*util.ToolCall{}
	for _, block := range resp.Content {
		// ContentBlockUnion has a Type field and direct fields
		if block.Type == "text" {
			responseText += block.Text
		} else if block.Type == "tool_use" {
			// Handle tool use blocks
			argsJSON := block.Input // Already json.RawMessage
			toolCalls = append(toolCalls, &util.ToolCall{
				Id:   block.ID,
				Type: "function",
				Function: util.FunctionCall{
					Name:       block.Name,
					Parameters: string(argsJSON),
				},
			})
		}
	}

	response := &util.CompletionResponse{
		Completion: responseText,
		ToolCalls:  toolCalls,
	}

	if a.verbose {
		LogCompletionResponse(*response, resp.ID)
	}

	return response, nil
}

// Anthropic streaming completion
func (a *AISDKLLM) completionStreamAnthropic(ctx context.Context, request *util.CompletionRequest, messages []aisdk.Message, writer io.Writer) (*util.CompletionResponse, error) {
	anthropicMessages, system, err := aisdk.MessagesToAnthropic(messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to Anthropic format: %w", err)
	}

	// Convert tools if provided
	var tools []anthropic.ToolUnionParam
	if len(request.Tools) > 0 {
		aisdkTools := convertToAISDKTools(request.Tools)
		tools = aisdk.ToolsToAnthropic(aisdkTools)
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(request.Model),
		Messages: anthropicMessages,
		System:   system,
		MaxTokens: int64(request.MaxTokens),
	}
	if len(tools) > 0 {
		params.Tools = tools
	}
	if request.Temperature > 0 {
		params.Temperature = anthropicparam.NewOpt(float64(request.Temperature))
	}

	if a.verbose {
		LogChatCompletionRequestAnthropic(params)
	}

	stream := a.provider.Anthropic.Messages.NewStreaming(ctx, params)
	dataStream := aisdk.AnthropicToDataStream(stream)

	var responseContent strings.Builder
	var toolCalls []*util.ToolCall

	// Process stream
	for part, err := range dataStream {
		if err != nil {
			return nil, err
		}

		switch p := part.(type) {
		case aisdk.TextStreamPart:
			writer.Write([]byte(p.Content))
			responseContent.WriteString(p.Content)

		case aisdk.ToolCallStartStreamPart:
			toolCalls = append(toolCalls, &util.ToolCall{
				Id:   p.ToolCallID,
				Type: "function",
				Function: util.FunctionCall{
					Name: p.ToolName,
				},
			})
			writer.Write([]byte(p.ToolName))
			writer.Write([]byte("("))

		case aisdk.ToolCallDeltaStreamPart:
			if len(toolCalls) > 0 {
				// Accumulate args delta
				writer.Write([]byte(p.ArgsTextDelta))
			}

		case aisdk.ToolCallStreamPart:
			if len(toolCalls) > 0 {
				argsJSON, _ := json.Marshal(p.Args)
				toolCalls[0].Function.Parameters = string(argsJSON)
			}
		}
	}

	if len(toolCalls) > 0 {
		writer.Write([]byte(")"))
	}
	fmt.Fprintf(writer, "\n")

	response := &util.CompletionResponse{
		Completion: responseContent.String(),
		ToolCalls:  toolCalls,
	}

	return response, nil
}

// Google completion (non-streaming)
func (a *AISDKLLM) completionGoogle(ctx context.Context, request *util.CompletionRequest, messages []aisdk.Message) (*util.CompletionResponse, error) {
	googleMessages, err := aisdk.MessagesToGoogle(messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to Google format: %w", err)
	}

	// Convert tools if provided
	var tools []*genai.Tool
	if len(request.Tools) > 0 {
		aisdkTools := convertToAISDKTools(request.Tools)
		convertedTools, err := aisdk.ToolsToGoogle(aisdkTools)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tools: %w", err)
		}
		tools = convertedTools
	}

	temp := request.Temperature
	config := &genai.GenerateContentConfig{
		MaxOutputTokens: int32(request.MaxTokens),
		Temperature:     &temp,
		Tools:           tools,
	}

	if a.verbose {
		LogChatCompletionRequestGoogle(request.Model, googleMessages, config)
	}

	var resp *genai.GenerateContentResponse
	err = withExponentialBackoff(func() error {
		var innerErr error
		resp, innerErr = a.provider.Google.Models.GenerateContent(ctx, request.Model, googleMessages, config)
		return innerErr
	})
	if err != nil {
		return nil, err
	}

	// Extract text content
	responseText := ""
	for _, candidate := range resp.Candidates {
		if candidate.Content != nil {
			for _, part := range candidate.Content.Parts {
				// Part is a struct, check its Text field directly
				if part.Text != "" {
					responseText += part.Text
				}
			}
		}
	}

	response := &util.CompletionResponse{
		Completion: responseText,
	}

	if a.verbose {
		LogCompletionResponse(*response, "")
	}

	return response, nil
}

// Google streaming completion
func (a *AISDKLLM) completionStreamGoogle(ctx context.Context, request *util.CompletionRequest, messages []aisdk.Message, writer io.Writer) (*util.CompletionResponse, error) {
	googleMessages, err := aisdk.MessagesToGoogle(messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to Google format: %w", err)
	}

	// Convert tools if provided
	var tools []*genai.Tool
	if len(request.Tools) > 0 {
		aisdkTools := convertToAISDKTools(request.Tools)
		convertedTools, err := aisdk.ToolsToGoogle(aisdkTools)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tools: %w", err)
		}
		tools = convertedTools
	}

	temp := request.Temperature
	config := &genai.GenerateContentConfig{
		MaxOutputTokens: int32(request.MaxTokens),
		Temperature:     &temp,
		Tools:           tools,
	}

	if a.verbose {
		LogChatCompletionRequestGoogle(request.Model, googleMessages, config)
	}

	stream := a.provider.Google.Models.GenerateContentStream(ctx, request.Model, googleMessages, config)
	dataStream := aisdk.GoogleToDataStream(stream)

	var responseContent strings.Builder

	// Process stream
	for part, err := range dataStream {
		if err != nil {
			return nil, err
		}

		switch p := part.(type) {
		case aisdk.TextStreamPart:
			writer.Write([]byte(p.Content))
			responseContent.WriteString(p.Content)
		}
	}

	fmt.Fprintf(writer, "\n")

	response := &util.CompletionResponse{
		Completion: responseContent.String(),
	}

	return response, nil
}

// OpenAI embeddings
func (a *AISDKLLM) embeddingsOpenAI(ctx context.Context, input []string, verbose bool) ([][]float32, error) {
	// OpenAI embeddings API accepts []string via union type
	req := openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: input,
		},
		Model: openai.EmbeddingModelTextEmbeddingAda002,
	}

	if verbose {
		log.Printf("Embedding %d strings", len(input))
	}

	var resp *openai.CreateEmbeddingResponse
	err := withExponentialBackoff(func() error {
		var innerErr error
		resp, innerErr = a.provider.OpenAI.Embeddings.New(ctx, req)
		return innerErr
	})
	if err != nil {
		return nil, err
	}

	// Convert []float64 to []float32
	result := make([][]float32, len(resp.Data))
	for i, embedding := range resp.Data {
		result[i] = make([]float32, len(embedding.Embedding))
		for j, val := range embedding.Embedding {
			result[i][j] = float32(val)
		}
	}

	return result, nil
}

// Helper function to convert util.ToolDefinition to aisdk.Tool
func convertToAISDKTools(tools []util.ToolDefinition) []aisdk.Tool {
	aisdkTools := make([]aisdk.Tool, len(tools))
	for i, tool := range tools {
		// Extract properties from the jsonschema.Definition and convert to map[string]any
		properties := make(map[string]any)
		if tool.Function.Parameters.Properties != nil {
			// Convert jsonschema.Definition map to map[string]any
			for key, def := range tool.Function.Parameters.Properties {
				// Marshal and unmarshal to convert jsonschema.Definition to map[string]any
				defJSON, err := json.Marshal(def)
				if err == nil {
					var defMap map[string]any
					if err := json.Unmarshal(defJSON, &defMap); err == nil {
						properties[key] = defMap
					} else {
						properties[key] = def
					}
				} else {
					properties[key] = def
				}
			}
		}
		
		aisdkTools[i] = aisdk.Tool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Schema: aisdk.Schema{
				Required:   tool.Function.Parameters.Required,
				Properties: properties,
			},
		}
	}
	return aisdkTools
}

// Helper function to convert jsonschema.Definition to map[string]any
func convertJSONSchemaToMap(schema interface{}) map[string]any {
	// Convert the schema to a map by marshaling and unmarshaling
	// This handles the jsonschema.Definition structure
	result := make(map[string]any)
	
	// Try to extract properties and required fields
	// The schema parameter should be a jsonschema.Definition
	if schemaMap, ok := schema.(map[string]interface{}); ok {
		return schemaMap
	}
	
	// If it's a struct, try to marshal/unmarshal
	jsonData, err := json.Marshal(schema)
	if err == nil {
		json.Unmarshal(jsonData, &result)
	}
	
	return result
}

// Note: withExponentialBackoff is defined in gpt.go and reused here

// Logging helpers (simplified versions - can be enhanced)
func LogChatCompletionRequestOpenAI(params openai.ChatCompletionNewParams) {
	// Simplified logging - can be enhanced to match existing format
	log.Printf("OpenAI Completion Request: model=%s, max_tokens=%v, temperature=%v",
		params.Model, params.MaxTokens, params.Temperature)
}

func LogChatCompletionRequestAnthropic(params anthropic.MessageNewParams) {
	log.Printf("Anthropic Completion Request: model=%s, max_tokens=%d",
		params.Model, params.MaxTokens)
}

func LogChatCompletionRequestGoogle(model string, messages []*genai.Content, config *genai.GenerateContentConfig) {
	log.Printf("Google Completion Request: model=%s, max_tokens=%d, temperature=%v",
		model, config.MaxOutputTokens, config.Temperature)
}

