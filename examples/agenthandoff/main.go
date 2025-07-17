package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	dotenv "github.com/joho/godotenv"
	swarmgo "github.com/prathyushnallamothu/swarmgo"
	"github.com/prathyushnallamothu/swarmgo/llm"
)

// Enhanced StreamHandler that logs agent transitions
type TestStreamHandler struct {
	*swarmgo.DefaultStreamHandler
}

func (h *TestStreamHandler) OnToken(token string) {
	fmt.Print(token)
}

func (h *TestStreamHandler) OnAgentTransition(fromAgent, toAgent *swarmgo.Agent, depth int) {
	fmt.Printf("\n\n🔄 [AGENT TRANSITION] %s → %s (depth: %d)\n", fromAgent.Name, toAgent.Name, depth)
}

func (h *TestStreamHandler) OnAgentReturn(fromAgent, toAgent *swarmgo.Agent, depth int) {
	fmt.Printf("\n\n↩️  [AGENT RETURN] %s → %s (depth: %d)\n", fromAgent.Name, toAgent.Name, depth)
}

func (h *TestStreamHandler) OnContextTransfer(context map[string]interface{}) {
	fmt.Printf("\n📦 [CONTEXT TRANSFER] Variables: %v\n", context)
}

func (h *TestStreamHandler) OnDepthLimitReached(maxDepth int) {
	fmt.Printf("\n⚠️  [DEPTH LIMIT] Maximum depth of %d reached\n", maxDepth)
}

func (h *TestStreamHandler) OnFunctionCallLimitReached(maxCalls int) {
	fmt.Printf("\n⚠️  [CALL LIMIT] Maximum function calls of %d reached\n", maxCalls)
}

func (h *TestStreamHandler) OnComplete(message llm.Message) {
	fmt.Printf("\n\n✅ [COMPLETE] Final message from %s\n", message.Name)
}

func (h *TestStreamHandler) OnError(err error) {
	fmt.Printf("\n❌ [ERROR] %v\n", err)
}

func transferToSpanishAgent(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	spanishAgent := &swarmgo.Agent{
		Name:         "SpanishAgent",
		Instructions: "You only speak Spanish. Always respond in Spanish. If you receive a request to transfer back to English, call the transferToEnglishAgent function.",
		Model:        "gpt-4",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "transferToEnglishAgent",
				Description: "Transfer back to the English-speaking agent.",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
				Function: func(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
					englishAgent := &swarmgo.Agent{
						Name:         "EnglishAgent",
						Instructions: "You only speak English. You can help with general questions and transfer Spanish-speaking users to the Spanish agent if needed.",
						Model:        "gpt-4",
					}
					return swarmgo.Result{
						Agent: englishAgent,
						Data:  "Transferring back to English Agent.",
					}
				},
			},
		},
	}

	// Add user preference to context
	contextVariables["preferred_language"] = "spanish"

	return swarmgo.Result{
		Agent: spanishAgent,
		Data:  "Transferring to Spanish Agent.",
	}
}

func transferToFrenchAgent(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	frenchAgent := &swarmgo.Agent{
		Name:         "FrenchAgent",
		Instructions: "You only speak French. Always respond in French. You are helpful and can answer questions about France and French culture.",
		Model:        "gpt-4",
	}

	// Add user preference to context
	contextVariables["preferred_language"] = "french"

	return swarmgo.Result{
		Agent: frenchAgent,
		Data:  "Transferring to French Agent.",
	}
}

func main() {
	dotenv.Load()

	client := swarmgo.NewSwarm(os.Getenv("OPENAI_API_KEY"), llm.OpenAI)

	englishAgent := &swarmgo.Agent{
		Name:         "EnglishAgent",
		Instructions: "You only speak English. You are a helpful language coordinator. When users speak in other languages or request language-specific help, transfer them to the appropriate agent.",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "transferToSpanishAgent",
				Description: "Transfer Spanish-speaking users to the Spanish agent.",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
				Function: transferToSpanishAgent,
			},
			{
				Name:        "transferToFrenchAgent",
				Description: "Transfer French-speaking users to the French agent.",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
				Function: transferToFrenchAgent,
			},
		},
		Model: "gpt-4",
	}

	// Create custom streaming limits for testing
	limits := swarmgo.StreamingLimits{
		MaxHandoffDepth:  2,
		MaxFunctionCalls: 10, // Lower for testing
	}

	// Create test scenarios
	testScenarios := []struct {
		name     string
		messages []llm.Message
		context  map[string]interface{}
	}{
		{
			name: "Spanish Hand-off",
			messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hola. ¿Cómo estás? I need help in Spanish."},
			},
			context: map[string]interface{}{
				"user_id": "test_123",
			},
		},
		{
			name: "French Hand-off",
			messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Bonjour! Je voudrais de l'aide en français."},
			},
			context: map[string]interface{}{
				"user_id": "test_456",
			},
		},
		{
			name: "Multi-level Hand-off",
			messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hola! Can you help me and then transfer me back to English?"},
			},
			context: map[string]interface{}{
				"user_id": "test_789",
			},
		},
	}

	for i, scenario := range testScenarios {
		fmt.Printf("\n" + strings.Repeat("=", 60) + "\n")
		fmt.Printf("🧪 TEST SCENARIO %d: %s\n", i+1, scenario.name)
		fmt.Printf(strings.Repeat("=", 60) + "\n")

		handler := &TestStreamHandler{
			DefaultStreamHandler: &swarmgo.DefaultStreamHandler{},
		}

		ctx := context.Background()

		fmt.Printf("👤 User: %s\n", scenario.messages[0].Content)
		fmt.Printf("🤖 Assistant: ")

		err := client.StreamingResponseWithLimits(
			ctx,
			englishAgent,
			scenario.messages,
			scenario.context,
			"", // no model override
			handler,
			limits,
			true, // debug mode
		)

		if err != nil {
			log.Printf("❌ Error in scenario %d: %v", i+1, err)
		}

		fmt.Printf("\n" + strings.Repeat("-", 60) + "\n")
		fmt.Printf("✅ Scenario %d completed\n", i+1)
	}

	fmt.Printf("\n" + strings.Repeat("=", 60) + "\n")
	fmt.Printf("🎉 All test scenarios completed!\n")
	fmt.Printf(strings.Repeat("=", 60) + "\n")
}
