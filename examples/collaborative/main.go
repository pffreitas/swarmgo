package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	dotenv "github.com/joho/godotenv"
	swarmgo "github.com/pffreitas/swarmgo"
	"github.com/pffreitas/swarmgo/llm"
)

func printStepResult(step swarmgo.StepResult) {
	fmt.Printf("\n\033[95mStep %d Results:\033[0m\n", step.StepNumber)
	fmt.Printf("Agent: %s\n", step.AgentName)
	fmt.Printf("Duration: %v\n", step.EndTime.Sub(step.StartTime))
	if step.Error != nil {
		fmt.Printf("\033[91mError: %v\033[0m\n", step.Error)
		return
	}

	fmt.Println("\nOutput:")
	for _, msg := range step.Output {
		switch msg.Role {
		case llm.RoleUser:
			fmt.Printf("\033[92m[User]\033[0m: %s\n", msg.Content)
		case llm.RoleAssistant:
			name := msg.Name
			if name == "" {
				name = "Assistant"
			}
			fmt.Printf("\033[94m[%s]\033[0m: %s\n", name, msg.Content)
		case llm.RoleFunction, "tool":
			fmt.Printf("\033[95m[Function Result]\033[0m: %s\n", msg.Content)
		}
	}

	if step.NextAgent != "" {
		fmt.Printf("\nNext Agent: %s\n", step.NextAgent)
	}
	fmt.Println("-----------------------------------------")
}

func main() {
	// Initial content for the document
	initialContent := `Introduction: SwarmGo is a Go package that allows you to create AI agents capable of interacting, coordinating, and executing tasks. Inspired by OpenAI's Swarm framework, SwarmGo focuses on making agent coordination and execution lightweight, highly controllable, and easily testable.

It achieves this through two primitive abstractions: Agents and handoffs. An Agent encompasses instructions and tools (functions it can execute), and can at any point choose to hand off a conversation to another Agent.

These primitives are powerful enough to express rich dynamics between tools and networks of agents, allowing you to build scalable, real-world solutions while avoiding a steep learning curve.
`

	// Load environment variables
	if err := dotenv.Load(); err != nil {
		log.Printf("Warning: .env file not found: %v", err)
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}

	// Create workflow with collaborative type
	workflow := swarmgo.NewWorkflow(apiKey, llm.OpenAI, swarmgo.CollaborativeWorkflow)

	// Set up cycle handling
	workflow.SetCycleHandling(swarmgo.ContinueOnCycle)
	workflow.SetCycleCallback(func(from, to string) (bool, error) {
		fmt.Printf("\n\033[93mCycle detected: %s -> %s\033[0m\n", from, to)
		fmt.Print("Do you want to continue the cycle? (y/n): ")
		var response string
		fmt.Scanln(&response)
		return strings.ToLower(response) == "y", nil
	})

	// Initialize document team agents
	editor := &swarmgo.Agent{
		Name: "editor",
		Instructions: `You are the editor agent. Your role is to:
1. Improve document structure and readability
2. Enhance clarity and engagement
3. Fix any grammatical or stylistic issues
4. Send your edits to the reviewer with "route to reviewer"
Focus on making concrete improvements, not delegating tasks.`,
		Model: "gpt-4",
	}

	reviewer := &swarmgo.Agent{
		Name: "reviewer",
		Instructions: `You are the reviewer agent. Your role is to:
1. Review the edited content for technical accuracy
2. Verify that all concepts are clearly explained
3. Suggest specific improvements
4. Route to writer with "route to writer" when review is complete
Provide direct feedback, not task assignments.`,
		Model: "gpt-4",
	}

	writer := &swarmgo.Agent{
		Name: "writer",
		Instructions: `You are the writer agent. Your role is to:
1. Implement suggested changes from the reviewer
2. Ensure consistent tone and style
3. Produce the final version of the document
4. Return to editor with "route to editor" for final review
Make direct changes to the content, don't delegate tasks.`,
		Model: "gpt-4",
	}

	// Add agents to document team
	workflow.AddAgentToTeam(editor, swarmgo.DocumentTeam)
	workflow.AddAgentToTeam(reviewer, swarmgo.DocumentTeam)
	workflow.AddAgentToTeam(writer, swarmgo.DocumentTeam)

	// Set editor as document team leader
	if err := workflow.SetTeamLeader(editor.Name, swarmgo.DocumentTeam); err != nil {
		log.Fatal("Error setting team leader:", err)
	}

	// Connect agents in collaborative pattern
	workflow.ConnectAgents(editor.Name, reviewer.Name)
	workflow.ConnectAgents(reviewer.Name, writer.Name)
	workflow.ConnectAgents(writer.Name, editor.Name)

	// Define user request with the initial content
	userRequest := fmt.Sprintf("Please improve this document through collaborative editing: %s", initialContent)

	// Execute workflow starting with the editor (team leader)
	fmt.Println("\n\033[96mStarting Collaborative Workflow\033[0m")
	fmt.Println("================================")

	result, err := workflow.Execute(editor.Name, userRequest)
	if err != nil {
		log.Fatal("Error executing workflow:", err)
	}

	// Print workflow summary
	fmt.Printf("\n\033[96mWorkflow Summary\033[0m\n")
	fmt.Printf("Total Duration: %v\n", result.EndTime.Sub(result.StartTime))
	fmt.Printf("Total Steps: %d\n", len(result.Steps))

	// Print detailed step results
	fmt.Println("\n\033[96mDetailed Step Results\033[0m")
	for _, step := range result.Steps {
		printStepResult(step)
	}

	// Extract final document version
	fmt.Println("\n\033[96mFinal Document Version\033[0m")
	var finalDoc string
	for i := len(result.Steps) - 1; i >= 0; i-- {
		step := result.Steps[i]
		if step.AgentName == writer.Name {
			for _, msg := range step.Output {
				if msg.Role == llm.RoleAssistant && strings.Contains(msg.Content, "FINAL:") {
					finalDoc = strings.TrimPrefix(msg.Content, "FINAL:")
					break
				}
			}
			if finalDoc != "" {
				break
			}
		}
	}

	if finalDoc != "" {
		fmt.Printf("\nFinal Document:\n%s\n", finalDoc)
	} else {
		fmt.Println("No final document version found")
	}
}
