package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"

	"github.com/invopop/jsonschema"
)

// ClaudeLLM implements the LLM interface for Anthropic's Claude
type ClaudeLLM struct {
	client *anthropic.Client
}

// NewClaudeLLM creates a new Claude LLM client
func NewClaudeLLM(apiKey string) *ClaudeLLM {
	client := anthropic.NewClient(option.WithAPIKey(apiKey))

	return &ClaudeLLM{client: &client}
}

// convertToClaudeMessages converts our generic Message type to Claude's message format
func convertToClaudeMessages(messages []Message) []anthropic.MessageParam {
	var claudeMessages []anthropic.MessageParam

	// First pass: collect tool calls and their results
	toolCallMap := make(map[string]string) // map[tool_id]result
	for _, msg := range messages {
		if msg.Role == RoleFunction {
			toolCallMap[msg.Name] = msg.Content
		}
	}

	for i, msg := range messages {
		switch msg.Role {
		case RoleSystem:
			// Claude handles system messages differently - we'll add it as a system prompt
			continue
		case RoleUser:
			claudeMessages = append(claudeMessages, anthropic.NewUserMessage(anthropic.NewTextBlock(msg.Content)))
		case RoleAssistant:
			// Skip assistant messages that are the last message when there are tool calls
			if len(msg.ToolCalls) > 0 && i == len(messages)-1 {
				continue
			}

			// If there are tool calls, we need to split this into multiple messages
			if len(msg.ToolCalls) > 0 {
				// First message: just the text content if any
				if msg.Content != "" {
					claudeMessages = append(claudeMessages, anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.Content)))
				}

				// Then for each tool call, create a tool use message
				for _, tc := range msg.ToolCalls {
					var args interface{}
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err == nil {
						// Create the tool use message with proper content block
						toolUseBlock := &anthropic.ToolUseBlockParam{
							ID:    tc.ID,
							Name:  tc.Function.Name,
							Input: args,
						}
						toolMsg := anthropic.NewAssistantMessage(anthropic.ContentBlockParamUnion{
							OfToolUse: toolUseBlock,
						})
						claudeMessages = append(claudeMessages, toolMsg)

						// Add the tool result immediately after if available
						if result, ok := toolCallMap[tc.Function.Name]; ok {
							toolResult := anthropic.NewUserMessage(
								anthropic.NewToolResultBlock(tc.ID, result, false))
							claudeMessages = append(claudeMessages, toolResult)
						}
					}
				}
			} else {
				// No tool calls, just add the content
				claudeMessages = append(claudeMessages, anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.Content)))
			}
		case RoleFunction:
			// Function messages are handled with tool calls
			continue
		}
	}

	return claudeMessages
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// generateJSONSchema generates a JSON schema for a given type
func generateJSONSchema[T any]() interface{} {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	return reflector.Reflect(v)
}

// convertToClaudeTools converts our generic Tool type to Claude's tool format
func convertToClaudeTools(tools []Tool) []anthropic.ToolUnionParam {
	if len(tools) == 0 {
		return nil
	}

	claudeTools := make([]anthropic.ToolUnionParam, len(tools))
	for i, tool := range tools {
		// Handle parameters based on their type
		var schema interface{}
		var params interface{} = tool.Function.Parameters
		switch params := params.(type) {
		case string:
			// If it's already a JSON string, parse it
			var parsedSchema interface{}
			if err := json.Unmarshal([]byte(params), &parsedSchema); err == nil {
				schema = parsedSchema
			} else {
				schema = params
			}
		case map[string]interface{}:
			// For maps, use directly since they should already be in schema format
			schema = params
		default:
			// For any other type, generate a schema using reflection
			schemaObj := generateJSONSchema[interface{}]()
			schema = schemaObj
		}

		// Create tool parameter with proper input schema
		toolParam := anthropic.ToolParam{
			Name:        tool.Function.Name,
			Description: anthropic.String(tool.Function.Description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: schema,
			},
		}

		claudeTools[i] = anthropic.ToolUnionParam{
			OfTool: &toolParam,
		}
	}
	return claudeTools
}

// convertFromClaudeMessage converts Claude's message type to our generic Message type
func convertFromClaudeMessage(msg anthropic.Message) Message {
	var content string
	var toolCalls []ToolCall

	for _, block := range msg.Content {
		switch block := block.AsAny().(type) {
		case anthropic.TextBlock:
			content = block.Text
		case anthropic.ToolUseBlock:
			argsBytes, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: ToolCallFunction{
					Name:      block.Name,
					Arguments: string(argsBytes),
				},
			})
		}
	}

	return Message{
		Role:      RoleAssistant,
		Content:   content,
		ToolCalls: toolCalls,
	}
}

// CreateChatCompletion implements the LLM interface for Claude
func (c *ClaudeLLM) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
	// Extract system message if present
	var systemPrompt string
	var nonSystemMessages []Message

	for _, msg := range req.Messages {
		if msg.Role == RoleSystem {
			systemPrompt = msg.Content
		} else {
			nonSystemMessages = append(nonSystemMessages, msg)
		}
	}

	// Convert all non-system messages at once
	messages := convertToClaudeMessages(nonSystemMessages)

	if req.MaxTokens == 0 {
		req.MaxTokens = 8192
	}

	// Create Claude request
	claudeReq := anthropic.MessageNewParams{
		Model:     anthropic.Model(req.Model),
		MaxTokens: int64(req.MaxTokens),
		Messages:  messages,
		Tools:     convertToClaudeTools(req.Tools),
	}

	if systemPrompt != "" {
		claudeReq.System = []anthropic.TextBlockParam{
			{Text: systemPrompt},
		}
	}

	if req.Temperature > 0 {
		claudeReq.Temperature = anthropic.Float(float64(req.Temperature))
	}

	// Make request to Claude API
	resp, err := c.client.Messages.New(ctx, claudeReq)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("claude API error: %v", err)
	}

	// Convert response
	message := convertFromClaudeMessage(*resp)

	return ChatCompletionResponse{
		ID: resp.ID,
		Choices: []Choice{{
			Index:        0,
			Message:      message,
			FinishReason: "stop", // Claude doesn't provide this explicitly
		}},
		Usage: Usage{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
	}, nil
}

// CreateChatCompletionStream implements the LLM interface for Claude streaming
func (c *ClaudeLLM) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (ChatCompletionStream, error) {
	// Extract system message if present
	var systemPrompt string
	var nonSystemMessages []Message

	for _, msg := range req.Messages {
		if msg.Role == RoleSystem {
			systemPrompt = msg.Content
		} else {
			nonSystemMessages = append(nonSystemMessages, msg)
		}
	}

	// Convert all non-system messages at once
	messages := convertToClaudeMessages(nonSystemMessages)

	if req.MaxTokens == 0 {
		req.MaxTokens = 8192
	}

	// Create Claude streaming request
	claudeReq := anthropic.MessageNewParams{
		Model:     anthropic.Model(req.Model),
		MaxTokens: int64(req.MaxTokens),
		Messages:  messages,
		Tools:     convertToClaudeTools(req.Tools),
	}

	if systemPrompt != "" {
		claudeReq.System = []anthropic.TextBlockParam{
			{Text: systemPrompt},
		}
	}

	if req.Temperature > 0 {
		claudeReq.Temperature = anthropic.Float(float64(req.Temperature))
	}

	// Create streaming response
	stream := c.client.Messages.NewStreaming(ctx, claudeReq)

	return &claudeStreamWrapper{
		stream:          stream,
		message:         anthropic.Message{},
		currentToolCall: nil,
		currentContent:  "",
	}, nil
}

// claudeStreamWrapper wraps Claude's stream to implement our ChatCompletionStream interface
type claudeStreamWrapper struct {
	stream          *ssestream.Stream[anthropic.MessageStreamEventUnion]
	message         anthropic.Message
	currentToolCall *ToolCall
	currentContent  string
}

func (w *claudeStreamWrapper) Recv() (ChatCompletionResponse, error) {
	if !w.stream.Next() {
		if err := w.stream.Err(); err != nil {
			return ChatCompletionResponse{}, err
		}
		return ChatCompletionResponse{}, io.EOF
	}

	event := w.stream.Current()
	err := w.message.Accumulate(event)
	if err != nil {
		return ChatCompletionResponse{}, err
	}

	message := Message{
		Role:    RoleAssistant,
		Content: w.currentContent,
	}

	switch event := event.AsAny().(type) {
	case anthropic.ContentBlockStartEvent:
		// Check if this is a tool use block by trying to cast the content block
		if toolUseBlock, ok := event.ContentBlock.AsAny().(anthropic.ToolUseBlock); ok {
			w.currentToolCall = &ToolCall{
				ID:   toolUseBlock.ID,
				Type: "function",
				Function: ToolCallFunction{
					Name: toolUseBlock.Name,
				},
			}
		}
	case anthropic.ContentBlockDeltaEvent:
		switch delta := event.Delta.AsAny().(type) {
		case anthropic.TextDelta:
			w.currentContent += delta.Text
			message.Content = w.currentContent
		case anthropic.InputJSONDelta:
			if w.currentToolCall != nil {
				if w.currentToolCall.Function.Arguments == "" {
					w.currentToolCall.Function.Arguments = delta.PartialJSON
				} else {
					w.currentToolCall.Function.Arguments += delta.PartialJSON
				}
			}
		}
	case anthropic.ContentBlockStopEvent:
		if w.currentToolCall != nil {
			message.ToolCalls = []ToolCall{*w.currentToolCall}
			w.currentToolCall = nil
		}
	case anthropic.MessageStopEvent:
		// Final message, include any pending tool calls
		if w.currentToolCall != nil {
			message.ToolCalls = []ToolCall{*w.currentToolCall}
			w.currentToolCall = nil
		}
	}

	return ChatCompletionResponse{
		ID: w.message.ID,
		Choices: []Choice{{
			Index:        0,
			Message:      message,
			FinishReason: string(event.Type),
		}},
	}, nil
}

func (w *claudeStreamWrapper) Close() error {
	w.stream.Close()
	return nil
}
