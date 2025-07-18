package swarmgo

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/pffreitas/swarmgo/llm"
)

// StreamHandler represents a handler for streaming responses with enhanced agent support
type StreamHandler interface {
	OnStart()
	OnToken(token string)
	OnToolCall(toolCall llm.ToolCall)
	OnComplete(message llm.Message)
	OnError(err error)

	OnAgentTransition(fromAgent, toAgent *Agent, depth int)
	OnAgentReturn(fromAgent, toAgent *Agent, depth int)
	OnContextTransfer(context map[string]interface{})
	OnDepthLimitReached(maxDepth int)
	OnFunctionCallLimitReached(maxCalls int)
}

// DefaultStreamHandler provides a basic implementation of StreamHandler
type DefaultStreamHandler struct{}

func (h *DefaultStreamHandler) OnStart()                                               {}
func (h *DefaultStreamHandler) OnToken(token string)                                   {}
func (h *DefaultStreamHandler) OnToolCall(toolCall llm.ToolCall)                       {}
func (h *DefaultStreamHandler) OnComplete(message llm.Message)                         {}
func (h *DefaultStreamHandler) OnError(err error)                                      {}
func (h *DefaultStreamHandler) OnAgentTransition(fromAgent, toAgent *Agent, depth int) {}
func (h *DefaultStreamHandler) OnAgentReturn(fromAgent, toAgent *Agent, depth int)     {}
func (h *DefaultStreamHandler) OnContextTransfer(context map[string]interface{})       {}
func (h *DefaultStreamHandler) OnDepthLimitReached(maxDepth int)                       {}
func (h *DefaultStreamHandler) OnFunctionCallLimitReached(maxCalls int)                {}

// StreamingLimits defines the limits for streaming operations
type StreamingLimits struct {
	MaxHandoffDepth  int // Maximum depth for agent hand-offs (default: 2)
	MaxFunctionCalls int // Maximum total function calls (default: 25)
}

// DefaultStreamingLimits returns the default streaming limits
func DefaultStreamingLimits() StreamingLimits {
	return StreamingLimits{
		MaxHandoffDepth:  2,
		MaxFunctionCalls: 25,
	}
}

// StreamContext represents the current streaming context
type StreamContext struct {
	Agent             *Agent
	Messages          []llm.Message
	ContextVariables  map[string]interface{}
	HandoffDepth      int
	FunctionCallCount int
	ParentContext     *StreamContext // For returning from hand-offs
}

// HandoffContext represents context for agent hand-offs
type HandoffContext struct {
	TargetAgent      *Agent
	TransferData     interface{}
	ContextVariables map[string]interface{}
	ReturnToParent   bool
}

// ToolCallExecution represents a tool call being executed
type ToolCallExecution struct {
	ID         string
	ToolCall   llm.ToolCall
	Function   *AgentFunction
	Arguments  map[string]interface{}
	IsComplete bool
	Result     Result
}

// HandoffManager manages agent transitions and depth tracking
type HandoffManager struct {
	limits       StreamingLimits
	currentDepth int
	callCount    int
	contextStack []*StreamContext
}

// NewHandoffManager creates a new HandoffManager
func NewHandoffManager(limits StreamingLimits) *HandoffManager {
	return &HandoffManager{
		limits:       limits,
		currentDepth: 0,
		callCount:    0,
		contextStack: make([]*StreamContext, 0),
	}
}

// CanHandoff checks if a hand-off is allowed
func (hm *HandoffManager) CanHandoff() bool {
	return hm.currentDepth < hm.limits.MaxHandoffDepth
}

// CanExecuteFunction checks if another function call is allowed
func (hm *HandoffManager) CanExecuteFunction() bool {
	return hm.callCount < hm.limits.MaxFunctionCalls
}

// PushContext adds a new context to the stack
func (hm *HandoffManager) PushContext(ctx *StreamContext) error {
	if !hm.CanHandoff() {
		return fmt.Errorf("maximum hand-off depth of %d reached", hm.limits.MaxHandoffDepth)
	}

	hm.contextStack = append(hm.contextStack, ctx)
	hm.currentDepth++
	return nil
}

// PopContext removes the current context from the stack
func (hm *HandoffManager) PopContext() *StreamContext {
	if len(hm.contextStack) == 0 {
		return nil
	}

	ctx := hm.contextStack[len(hm.contextStack)-1]
	hm.contextStack = hm.contextStack[:len(hm.contextStack)-1]
	hm.currentDepth--
	return ctx
}

// CurrentContext returns the current context
func (hm *HandoffManager) CurrentContext() *StreamContext {
	if len(hm.contextStack) == 0 {
		return nil
	}
	return hm.contextStack[len(hm.contextStack)-1]
}

// IncrementCallCount increments the function call counter
func (hm *HandoffManager) IncrementCallCount() {
	hm.callCount++
}

// GetCallCount returns the current function call count
func (hm *HandoffManager) GetCallCount() int {
	return hm.callCount
}

// GetDepth returns the current hand-off depth
func (hm *HandoffManager) GetDepth() int {
	return hm.currentDepth
}

// ToolCallManager manages tool call execution and tracking
type ToolCallManager struct {
	inProgress map[string]*ToolCallExecution
	processed  map[string]bool
}

// NewToolCallManager creates a new ToolCallManager
func NewToolCallManager() *ToolCallManager {
	return &ToolCallManager{
		inProgress: make(map[string]*ToolCallExecution),
		processed:  make(map[string]bool),
	}
}

// ProcessToolCallDelta processes a streaming tool call delta
func (tcm *ToolCallManager) ProcessToolCallDelta(toolCall llm.ToolCall, agentFunctions []AgentFunction) (*ToolCallExecution, bool) {
	if toolCall.ID == "" {
		return nil, false
	}

	// Skip if already processed
	if tcm.processed[toolCall.ID] {
		return nil, false
	}

	// Get or create execution
	execution, exists := tcm.inProgress[toolCall.ID]
	if !exists {
		execution = &ToolCallExecution{
			ID: toolCall.ID,
			ToolCall: llm.ToolCall{
				ID:   toolCall.ID,
				Type: toolCall.Type,
				Function: llm.ToolCallFunction{
					Name:      toolCall.Function.Name,
					Arguments: "",
				},
			},
			IsComplete: false,
		}

		// Find the corresponding function
		for _, af := range agentFunctions {
			if af.Name == toolCall.Function.Name {
				execution.Function = &af
				break
			}
		}

		tcm.inProgress[toolCall.ID] = execution
	}

	// Update function name if provided
	if toolCall.Function.Name != "" && execution.ToolCall.Function.Name == "" {
		execution.ToolCall.Function.Name = toolCall.Function.Name

		// Find the corresponding function if not already found
		if execution.Function == nil {
			for _, af := range agentFunctions {
				if af.Name == toolCall.Function.Name {
					execution.Function = &af
					break
				}
			}
		}
	}

	// Accumulate arguments
	if toolCall.Function.Arguments != "" {
		execution.ToolCall.Function.Arguments += toolCall.Function.Arguments
	}

	// Try to parse arguments to check if complete
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(execution.ToolCall.Function.Arguments), &args); err == nil {
		execution.Arguments = args
		execution.IsComplete = true
		return execution, true
	}

	return execution, false
}

// MarkAsProcessed marks a tool call as processed
func (tcm *ToolCallManager) MarkAsProcessed(toolCallID string) {
	tcm.processed[toolCallID] = true
	delete(tcm.inProgress, toolCallID)
}

// StreamOrchestrator is the main coordinator for streaming operations
type StreamOrchestrator struct {
	client      llm.LLM
	handoffMgr  *HandoffManager
	toolCallMgr *ToolCallManager
	handler     StreamHandler
	debug       bool
}

// NewStreamOrchestrator creates a new StreamOrchestrator
func NewStreamOrchestrator(client llm.LLM, limits StreamingLimits, handler StreamHandler, debug bool) *StreamOrchestrator {
	if handler == nil {
		handler = &DefaultStreamHandler{}
	}

	return &StreamOrchestrator{
		client:      client,
		handoffMgr:  NewHandoffManager(limits),
		toolCallMgr: NewToolCallManager(),
		handler:     handler,
		debug:       debug,
	}
}

// ExecuteToolCall executes a single tool call and handles potential hand-offs
func (so *StreamOrchestrator) ExecuteToolCall(ctx context.Context, execution *ToolCallExecution, streamCtx *StreamContext) (*HandoffContext, error) {
	if execution.Function == nil {
		return nil, fmt.Errorf("unknown function: %s", execution.ToolCall.Function.Name)
	}

	if !so.handoffMgr.CanExecuteFunction() {
		so.handler.OnFunctionCallLimitReached(so.handoffMgr.limits.MaxFunctionCalls)
		return nil, fmt.Errorf("maximum function call limit of %d reached", so.handoffMgr.limits.MaxFunctionCalls)
	}

	if so.debug {
		fmt.Printf("Debug: Executing function %s with args: %v\n", execution.Function.Name, execution.Arguments)
	}

	// Execute the function
	result := execution.Function.Function(execution.Arguments, streamCtx.ContextVariables)
	execution.Result = result

	// Increment call count
	so.handoffMgr.IncrementCallCount()

	// Check for agent hand-off
	if result.Agent != nil {
		if !so.handoffMgr.CanHandoff() {
			so.handler.OnDepthLimitReached(so.handoffMgr.limits.MaxHandoffDepth)
			return nil, fmt.Errorf("maximum hand-off depth of %d reached", so.handoffMgr.limits.MaxHandoffDepth)
		}

		// Create hand-off context
		handoffCtx := &HandoffContext{
			TargetAgent:      result.Agent,
			TransferData:     result.Data,
			ContextVariables: make(map[string]interface{}),
			ReturnToParent:   true, // Allow the handed-off agent to decide
		}

		// Copy context variables for transfer
		for k, v := range streamCtx.ContextVariables {
			handoffCtx.ContextVariables[k] = v
		}

		// Notify handler of context transfer
		so.handler.OnContextTransfer(handoffCtx.ContextVariables)

		return handoffCtx, nil
	}

	return nil, nil
}

// StreamingResponse handles streaming chat completions with agent hand-off support
func (s *Swarm) StreamingResponse(
	ctx context.Context,
	agent *Agent,
	messages []llm.Message,
	contextVariables map[string]interface{},
	modelOverride string,
	handler StreamHandler,
	debug bool,
) error {
	return s.StreamingResponseWithLimits(ctx, agent, messages, contextVariables, modelOverride, handler, DefaultStreamingLimits(), debug)
}

// StreamingResponseWithLimits handles streaming with custom limits
func (s *Swarm) StreamingResponseWithLimits(
	ctx context.Context,
	agent *Agent,
	messages []llm.Message,
	contextVariables map[string]interface{},
	modelOverride string,
	handler StreamHandler,
	limits StreamingLimits,
	debug bool,
) error {
	orchestrator := NewStreamOrchestrator(s.client, limits, handler, debug)

	// Create initial stream context
	initialContext := &StreamContext{
		Agent:             agent,
		Messages:          messages,
		ContextVariables:  contextVariables,
		HandoffDepth:      0,
		FunctionCallCount: 0,
		ParentContext:     nil,
	}

	return orchestrator.ProcessStreamContext(ctx, initialContext, modelOverride)
}

// ProcessStreamContext processes a single stream context (recursive for hand-offs)
func (so *StreamOrchestrator) ProcessStreamContext(ctx context.Context, streamCtx *StreamContext, modelOverride string) error {
	// Push context to stack
	if err := so.handoffMgr.PushContext(streamCtx); err != nil {
		so.handler.OnError(err)
		return err
	}
	defer so.handoffMgr.PopContext()

	if so.debug {
		fmt.Printf("Debug: Processing stream context for agent: %s (depth: %d)\n",
			streamCtx.Agent.Name, so.handoffMgr.GetDepth())
	}

	// Prepare the initial system message with agent instructions
	instructions := streamCtx.Agent.Instructions
	if streamCtx.Agent.InstructionsFunc != nil {
		instructions = streamCtx.Agent.InstructionsFunc(streamCtx.ContextVariables)
	}

	allMessages := append([]llm.Message{
		{
			Role:    llm.RoleSystem,
			Content: instructions,
		},
	}, streamCtx.Messages...)

	// Build tool definitions
	var tools []llm.Tool
	for _, af := range streamCtx.Agent.Functions {
		def := FunctionToDefinition(af)
		if so.debug {
			fmt.Printf("Debug: Adding tool: %s\n", def.Name)
		}
		tools = append(tools, llm.Tool{
			Type: "function",
			Function: &llm.Function{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  def.Parameters,
			},
		})
	}

	// Prepare the streaming request
	model := streamCtx.Agent.Model
	if modelOverride != "" {
		model = modelOverride
	}

	req := llm.ChatCompletionRequest{
		Model:    model,
		Messages: allMessages,
		Tools:    tools,
		Stream:   true,
	}

	stream, err := so.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		if so.debug {
			fmt.Printf("Debug: Stream creation error: %v\n", err)
		}
		so.handler.OnError(fmt.Errorf("failed to create chat completion stream: %v", err))
		return err
	}
	defer stream.Close()

	so.handler.OnStart()

	var currentMessage llm.Message
	currentMessage.Role = llm.RoleAssistant
	currentMessage.Name = streamCtx.Agent.Name

	for {
		select {
		case <-ctx.Done():
			so.handler.OnError(ctx.Err())
			return ctx.Err()
		default:
			response, err := stream.Recv()
			if err != nil {
				if err.Error() == "EOF" {
					so.handler.OnComplete(currentMessage)
					return nil
				}
				if so.debug {
					fmt.Printf("Debug: Error receiving from stream: %v\n", err)
				}
				so.handler.OnError(fmt.Errorf("error receiving from stream: %v", err))
				return err
			}

			if len(response.Choices) == 0 {
				continue
			}

			choice := response.Choices[0]

			// Handle content streaming
			if choice.Message.Content != "" {
				currentMessage.Content += choice.Message.Content
				so.handler.OnToken(choice.Message.Content)
			}

			// Handle tool calls
			if len(choice.Message.ToolCalls) > 0 {
				if err := so.processToolCalls(ctx, choice.Message.ToolCalls, streamCtx, &currentMessage, &allMessages, modelOverride); err != nil {
					so.handler.OnError(err)
					return err
				}
			}
		}
	}
}

// processToolCalls handles tool call processing and potential hand-offs
func (so *StreamOrchestrator) processToolCalls(
	ctx context.Context,
	toolCalls []llm.ToolCall,
	streamCtx *StreamContext,
	currentMessage *llm.Message,
	allMessages *[]llm.Message,
	modelOverride string,
) error {
	for _, toolCall := range toolCalls {
		if so.debug {
			fmt.Printf("Debug: Processing tool call: ID=%s Name=%s\n", toolCall.ID, toolCall.Function.Name)
		}

		// Process tool call delta
		execution, isComplete := so.toolCallMgr.ProcessToolCallDelta(toolCall, streamCtx.Agent.Functions)
		if execution == nil {
			continue
		}

		// If tool call is complete, execute it
		if isComplete {
			// Mark as processed
			so.toolCallMgr.MarkAsProcessed(execution.ID)

			// Add to current message
			currentMessage.ToolCalls = append(currentMessage.ToolCalls, execution.ToolCall)
			so.handler.OnToolCall(execution.ToolCall)

			// Execute the tool call
			handoffCtx, err := so.ExecuteToolCall(ctx, execution, streamCtx)
			if err != nil {
				// Create error function response message
				errorMsg := llm.Message{
					Role:    llm.RoleFunction,
					Content: fmt.Sprintf("Error: %v", err),
					Name:    execution.Function.Name,
				}
				*allMessages = append(*allMessages, *currentMessage, errorMsg)
				continue
			}

			// Handle normal function result
			var resultContent string
			if execution.Result.Error != nil {
				resultContent = fmt.Sprintf("Error: %v", execution.Result.Error)
			} else {
				resultContent = fmt.Sprintf("%v", execution.Result.Data)
			}

			functionMessage := llm.Message{
				Role:    llm.RoleFunction,
				Content: resultContent,
				Name:    execution.Function.Name,
			}

			*allMessages = append(*allMessages, *currentMessage, functionMessage)

			// Handle agent hand-off
			if handoffCtx != nil {
				if so.debug {
					fmt.Printf("Debug: Agent hand-off from %s to %s\n",
						streamCtx.Agent.Name, handoffCtx.TargetAgent.Name)
				}

				// Notify handler of agent transition
				so.handler.OnAgentTransition(streamCtx.Agent, handoffCtx.TargetAgent, so.handoffMgr.GetDepth())

				// Create fresh message context for handed-off agent with only the transfer data
				var newMessages []llm.Message
				if handoffCtx.TransferData != nil {
					transferContent := fmt.Sprintf("%v", handoffCtx.TransferData)
					newMessages = []llm.Message{
						{
							Role:    llm.RoleUser,
							Content: transferContent,
						},
					}
				}

				// Create new stream context for handed-off agent with cleared context
				newStreamCtx := &StreamContext{
					Agent:             handoffCtx.TargetAgent,
					Messages:          newMessages, // Fresh context with only transfer data
					ContextVariables:  handoffCtx.ContextVariables,
					HandoffDepth:      streamCtx.HandoffDepth + 1,
					FunctionCallCount: streamCtx.FunctionCallCount,
					ParentContext:     streamCtx,
				}

				// Process the handed-off agent (recursive call)
				if err := so.ProcessStreamContext(ctx, newStreamCtx, modelOverride); err != nil {
					// If handed-off agent errors, return error to original agent
					if so.debug {
						fmt.Printf("Debug: Handed-off agent error: %v\n", err)
					}
					return err
				}

				// Notify handler of agent return
				so.handler.OnAgentReturn(handoffCtx.TargetAgent, streamCtx.Agent, so.handoffMgr.GetDepth())

				if so.debug {
					fmt.Printf("Debug: Returned from hand-off to %s\n", streamCtx.Agent.Name)
				}
			}

			// Reset current message for next response
			*currentMessage = llm.Message{
				Role: llm.RoleAssistant,
				Name: streamCtx.Agent.Name,
			}
		}
	}

	return nil
}
