package butterfish

import (
	"encoding/json"

	"github.com/bakks/butterfish/util"
	"github.com/coder/aisdk-go"
)

// ConvertHistoryBlocksToAISDKMessages converts butterfish HistoryBlocks to aisdk-go Message format
func ConvertHistoryBlocksToAISDKMessages(systemMsg string, blocks []util.HistoryBlock) ([]aisdk.Message, error) {
	messages := []aisdk.Message{}

	// Add system message if provided
	if systemMsg != "" {
		messages = append(messages, aisdk.Message{
			Role:    "system",
			Content: systemMsg,
			Parts: []aisdk.Part{
				{
					Type: aisdk.PartTypeText,
					Text: systemMsg,
				},
			},
		})
	}

	// Convert history blocks
	for _, block := range blocks {
		if block.Content == "" && block.FunctionName == "" && block.ToolCalls == nil {
			// skip empty blocks
			continue
		}

		role := ShellHistoryTypeToRole(block.Type)
		msg := aisdk.Message{
			Role:    role,
			Content: block.Content,
			Parts:   []aisdk.Part{},
		}

		// Handle text content
		if block.Content != "" {
			msg.Parts = append(msg.Parts, aisdk.Part{
				Type: aisdk.PartTypeText,
				Text: block.Content,
			})
		}

		// Handle function calls (legacy - convert to tool invocations)
		if role == "assistant" && block.FunctionName != "" {
			// Parse function parameters
			var args map[string]any
			if block.FunctionParams != "" {
				if err := json.Unmarshal([]byte(block.FunctionParams), &args); err != nil {
					// If parsing fails, treat as empty args
					args = make(map[string]any)
				}
			} else {
				args = make(map[string]any)
			}

			msg.Parts = append(msg.Parts, aisdk.Part{
				Type: aisdk.PartTypeToolInvocation,
				ToolInvocation: &aisdk.ToolInvocation{
					State:      aisdk.ToolInvocationStateCall,
					ToolCallID: block.FunctionName, // Use function name as ID if no ID provided
					ToolName:   block.FunctionName,
					Args:       args,
				},
			})
		}

		// Handle tool calls
		if block.ToolCalls != nil && len(block.ToolCalls) > 0 {
			for _, toolCall := range block.ToolCalls {
				var args map[string]any
				if toolCall.Function.Parameters != "" {
					if err := json.Unmarshal([]byte(toolCall.Function.Parameters), &args); err != nil {
						args = make(map[string]any)
					}
				} else {
					args = make(map[string]any)
				}

				msg.Parts = append(msg.Parts, aisdk.Part{
					Type: aisdk.PartTypeToolInvocation,
					ToolInvocation: &aisdk.ToolInvocation{
						State:      aisdk.ToolInvocationStateCall,
						ToolCallID: toolCall.Id,
						ToolName:   toolCall.Function.Name,
						Args:       args,
					},
				})
			}
		}

		// Handle function/tool outputs (these become tool result messages)
		if role == "function" || role == "tool" {
			// Tool results are handled separately in the conversation flow
			// For now, we'll include them as text parts
			if block.Content != "" {
				msg.Parts = []aisdk.Part{
					{
						Type: aisdk.PartTypeText,
						Text: block.Content,
					},
				}
			}
		}

		messages = append(messages, msg)
	}

	return messages, nil
}

// ConvertCompletionRequestToAISDKMessages converts a CompletionRequest to aisdk-go Message slice
func ConvertCompletionRequestToAISDKMessages(request *util.CompletionRequest) ([]aisdk.Message, error) {
	messages, err := ConvertHistoryBlocksToAISDKMessages(request.SystemMessage, request.HistoryBlocks)
	if err != nil {
		return nil, err
	}

	// Add current prompt as user message if provided
	if request.Prompt != "" {
		messages = append(messages, aisdk.Message{
			Role:    "user",
			Content: request.Prompt,
			Parts: []aisdk.Part{
				{
					Type: aisdk.PartTypeText,
					Text: request.Prompt,
				},
			},
		})
	}

	return messages, nil
}

