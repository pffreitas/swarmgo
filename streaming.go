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

// AgentContext represents the context for a single agent in the stack
type AgentContext struct {
	Agent            *Agent
	Messages         []llm.Message
	ContextVariables map[string]interface{}
	Model            string
	CurrentMessage   llm.Message // Current message being built
}

// AgentStack manages the stack of agent contexts
type AgentStack struct {
	contexts []*AgentContext
	limits   StreamingLimits
}

// NewAgentStack creates a new agent stack
func NewAgentStack(limits StreamingLimits) *AgentStack {
	return &AgentStack{
		contexts: make([]*AgentContext, 0),
		limits:   limits,
	}
}

// Push adds a new agent context to the stack
func (as *AgentStack) Push(ctx *AgentContext) error {
	if len(as.contexts) >= as.limits.MaxHandoffDepth {
		return fmt.Errorf("maximum hand-off depth of %d reached", as.limits.MaxHandoffDepth)
	}
	as.contexts = append(as.contexts, ctx)
	return nil
}

// Pop removes the top agent context from the stack
func (as *AgentStack) Pop() *AgentContext {
	if len(as.contexts) == 0 {
		return nil
	}
	ctx := as.contexts[len(as.contexts)-1]
	as.contexts = as.contexts[:len(as.contexts)-1]
	return ctx
}

// Current returns the current (top) agent context
func (as *AgentStack) Current() *AgentContext {
	if len(as.contexts) == 0 {
		return nil
	}
	return as.contexts[len(as.contexts)-1]
}

// Depth returns the current stack depth
func (as *AgentStack) Depth() int {
	return len(as.contexts)
}

// IsEmpty returns true if the stack is empty
func (as *AgentStack) IsEmpty() bool {
	return len(as.contexts) == 0
}

// FindAgentInStack searches for an agent in the stack and returns its position (-1 if not found)
func (as *AgentStack) FindAgentInStack(targetAgent *Agent) int {
	for i := len(as.contexts) - 1; i >= 0; i-- {
		if as.contexts[i].Agent == targetAgent {
			return i
		}
	}
	return -1
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
	client            llm.LLM
	agentStack        *AgentStack
	toolCallMgr       *ToolCallManager
	handler           StreamHandler
	debug             bool
	functionCallCount int
}

// NewStreamOrchestrator creates a new StreamOrchestrator
func NewStreamOrchestrator(client llm.LLM, limits StreamingLimits, handler StreamHandler, debug bool) *StreamOrchestrator {
	if handler == nil {
		handler = &DefaultStreamHandler{}
	}

	return &StreamOrchestrator{
		client:            client,
		agentStack:        NewAgentStack(limits),
		toolCallMgr:       NewToolCallManager(),
		handler:           handler,
		debug:             debug,
		functionCallCount: 0,
	}
}

// ExecuteToolCall executes a single tool call and handles potential hand-offs
func (so *StreamOrchestrator) ExecuteToolCall(ctx context.Context, execution *ToolCallExecution, agentCtx *AgentContext) (*HandoffContext, error) {
	if execution.Function == nil {
		return nil, fmt.Errorf("unknown function: %s", execution.ToolCall.Function.Name)
	}

	if so.functionCallCount >= so.agentStack.limits.MaxFunctionCalls {
		so.handler.OnFunctionCallLimitReached(so.agentStack.limits.MaxFunctionCalls)
		return nil, fmt.Errorf("maximum function call limit of %d reached", so.agentStack.limits.MaxFunctionCalls)
	}

	if so.debug {
		fmt.Printf("Debug: Executing function %s with args: %v\n", execution.Function.Name, execution.Arguments)
	}

	// Execute the function
	result := execution.Function.Function(execution.Arguments, agentCtx.ContextVariables)
	execution.Result = result

	// Increment call count
	so.functionCallCount++

	// Check for agent hand-off
	if result.Agent != nil {
		// Create hand-off context
		handoffCtx := &HandoffContext{
			TargetAgent:      result.Agent,
			TransferData:     result.Data,
			ContextVariables: make(map[string]interface{}),
			ReturnToParent:   true,
		}

		// Copy context variables for transfer
		for k, v := range agentCtx.ContextVariables {
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

	// Create initial agent context
	model := agent.Model
	if modelOverride != "" {
		model = modelOverride
	}

	initialContext := &AgentContext{
		Agent:            agent,
		Messages:         messages,
		ContextVariables: contextVariables,
		Model:            model,
		CurrentMessage:   llm.Message{Role: llm.RoleAssistant, Name: agent.Name},
	}

	// Push initial context to stack
	if err := orchestrator.agentStack.Push(initialContext); err != nil {
		handler.OnError(err)
		return err
	}

	// Start the main processing loop
	return orchestrator.ProcessAgentStack(ctx)
}

// ProcessAgentStack processes the agent stack with stream creation inside the loop
func (so *StreamOrchestrator) ProcessAgentStack(ctx context.Context) error {
	so.handler.OnStart()

	// Main processing loop
	for !so.agentStack.IsEmpty() {
		// Get current agent context
		currentCtx := so.agentStack.Current()
		if currentCtx == nil {
			break
		}

		if so.debug {
			fmt.Printf("Debug: Processing agent: %s (stack depth: %d)\n",
				currentCtx.Agent.Name, so.agentStack.Depth())
		}

		// Prepare messages with agent instructions
		instructions := currentCtx.Agent.Instructions
		if currentCtx.Agent.InstructionsFunc != nil {
			instructions = currentCtx.Agent.InstructionsFunc(currentCtx.ContextVariables)
		}

		allMessages := append([]llm.Message{
			{
				Role:    llm.RoleSystem,
				Content: instructions,
			},
		}, currentCtx.Messages...)

		// Build tool definitions
		var tools []llm.Tool
		for _, af := range currentCtx.Agent.Functions {
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

		// Create streaming request
		req := llm.ChatCompletionRequest{
			Model:    currentCtx.Model,
			Messages: allMessages,
			Tools:    tools,
			Stream:   true,
		}

		// Create stream inside the loop
		stream, err := so.client.CreateChatCompletionStream(ctx, req)
		if err != nil {
			if so.debug {
				fmt.Printf("Debug: Stream creation error: %v\n", err)
			}
			so.handler.OnError(fmt.Errorf("failed to create chat completion stream: %v", err))
			return err
		}

		// Process this agent's stream
		agentHandoff, err := so.processAgentStream(ctx, stream, currentCtx)
		stream.Close()

		if err != nil {
			so.handler.OnError(err)
			return err
		}

		// Handle agent handoff if one occurred
		if agentHandoff != nil {
			if err := so.handleAgentHandoff(agentHandoff, currentCtx); err != nil {
				so.handler.OnError(err)
				return err
			}
		} else {
			// No handoff, complete this agent and pop from stack
			so.handler.OnComplete(currentCtx.CurrentMessage)
			so.agentStack.Pop()

			// If there's a parent agent, add the response to its context
			if !so.agentStack.IsEmpty() {
				parentCtx := so.agentStack.Current()
				parentCtx.Messages = append(parentCtx.Messages, currentCtx.CurrentMessage)
			}
		}
	}

	return nil
}

// processAgentStream processes a single agent's stream until completion or handoff
func (so *StreamOrchestrator) processAgentStream(ctx context.Context, stream llm.ChatCompletionStream, agentCtx *AgentContext) (*HandoffContext, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			response, err := stream.Recv()
			if err != nil {
				if err.Error() == "EOF" {
					// Stream completed normally
					return nil, nil
				}
				if so.debug {
					fmt.Printf("Debug: Error receiving from stream: %v\n", err)
				}
				return nil, fmt.Errorf("error receiving from stream: %v", err)
			}

			if len(response.Choices) == 0 {
				continue
			}

			choice := response.Choices[0]

			// Handle content streaming
			if choice.Message.Content != "" {
				agentCtx.CurrentMessage.Content += choice.Message.Content
				so.handler.OnToken(choice.Message.Content)
			}

			// Handle tool calls
			if len(choice.Message.ToolCalls) > 0 {
				handoff, err := so.processToolCallsInStream(ctx, choice.Message.ToolCalls, agentCtx)
				if err != nil {
					return nil, err
				}
				if handoff != nil {
					// Tool call resulted in handoff, return it
					return handoff, nil
				}
			}
		}
	}
}

// processToolCallsInStream processes tool calls within a stream
func (so *StreamOrchestrator) processToolCallsInStream(
	ctx context.Context,
	toolCalls []llm.ToolCall,
	agentCtx *AgentContext,
) (*HandoffContext, error) {
	for _, toolCall := range toolCalls {
		if so.debug {
			fmt.Printf("Debug: Processing tool call: ID=%s Name=%s\n", toolCall.ID, toolCall.Function.Name)
		}

		// Process tool call delta
		execution, isComplete := so.toolCallMgr.ProcessToolCallDelta(toolCall, agentCtx.Agent.Functions)
		if execution == nil {
			continue
		}

		// If tool call is complete, execute it
		if isComplete {
			// Mark as processed
			so.toolCallMgr.MarkAsProcessed(execution.ID)

			// Add to current message
			agentCtx.CurrentMessage.ToolCalls = append(agentCtx.CurrentMessage.ToolCalls, execution.ToolCall)
			so.handler.OnToolCall(execution.ToolCall)

			// Execute the tool call
			handoffCtx, err := so.ExecuteToolCall(ctx, execution, agentCtx)
			if err != nil {
				// Create error function response message
				errorMsg := llm.Message{
					Role:    llm.RoleFunction,
					Content: fmt.Sprintf("Error: %v", err),
					Name:    execution.Function.Name,
				}
				agentCtx.Messages = append(agentCtx.Messages, agentCtx.CurrentMessage, errorMsg)

				// Reset current message
				agentCtx.CurrentMessage = llm.Message{
					Role: llm.RoleAssistant,
					Name: agentCtx.Agent.Name,
				}
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

			agentCtx.Messages = append(agentCtx.Messages, agentCtx.CurrentMessage, functionMessage)

			// Handle agent hand-off
			if handoffCtx != nil {
				return handoffCtx, nil
			}

			// Reset current message for next response
			agentCtx.CurrentMessage = llm.Message{
				Role: llm.RoleAssistant,
				Name: agentCtx.Agent.Name,
			}
		}
	}

	return nil, nil
}

// handleAgentHandoff handles the logic for agent handoffs (stack manipulation)
func (so *StreamOrchestrator) handleAgentHandoff(handoffCtx *HandoffContext, currentCtx *AgentContext) error {
	// Check if the target agent is already in the stack (return to previous agent)
	targetPosition := so.agentStack.FindAgentInStack(handoffCtx.TargetAgent)

	if targetPosition >= 0 {
		// Return to previous agent - pop back to that position
		if so.debug {
			fmt.Printf("Debug: Returning to previous agent %s at position %d\n",
				handoffCtx.TargetAgent.Name, targetPosition)
		}

		// Add current response to the target agent's context
		targetCtx := so.agentStack.contexts[targetPosition]

		// Add the transfer data as a user message if present
		if handoffCtx.TransferData != nil {
			transferContent := fmt.Sprintf("%v", handoffCtx.TransferData)
			targetCtx.Messages = append(targetCtx.Messages, llm.Message{
				Role:    llm.RoleUser,
				Content: transferContent,
			})
		}

		// Update context variables
		for k, v := range handoffCtx.ContextVariables {
			targetCtx.ContextVariables[k] = v
		}

		// Pop all agents above the target
		for so.agentStack.Depth() > targetPosition+1 {
			poppedCtx := so.agentStack.Pop()
			so.handler.OnAgentReturn(poppedCtx.Agent, targetCtx.Agent, so.agentStack.Depth())
		}

	} else {
		// New agent handoff - push to stack
		if so.debug {
			fmt.Printf("Debug: Agent hand-off from %s to %s\n",
				currentCtx.Agent.Name, handoffCtx.TargetAgent.Name)
		}

		// Check depth limits
		if so.agentStack.Depth() >= so.agentStack.limits.MaxHandoffDepth {
			so.handler.OnDepthLimitReached(so.agentStack.limits.MaxHandoffDepth)
			return fmt.Errorf("maximum hand-off depth of %d reached", so.agentStack.limits.MaxHandoffDepth)
		}

		// Notify handler of agent transition
		so.handler.OnAgentTransition(currentCtx.Agent, handoffCtx.TargetAgent, so.agentStack.Depth())

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

		// Determine model for new agent
		model := handoffCtx.TargetAgent.Model
		if currentCtx.Model != currentCtx.Agent.Model {
			// Preserve model override if it was set
			model = currentCtx.Model
		}

		// Create new agent context
		newAgentCtx := &AgentContext{
			Agent:            handoffCtx.TargetAgent,
			Messages:         newMessages, // Fresh context with only transfer data
			ContextVariables: handoffCtx.ContextVariables,
			Model:            model,
			CurrentMessage:   llm.Message{Role: llm.RoleAssistant, Name: handoffCtx.TargetAgent.Name},
		}

		// Push new agent to stack
		if err := so.agentStack.Push(newAgentCtx); err != nil {
			return err
		}
	}

	return nil
}
