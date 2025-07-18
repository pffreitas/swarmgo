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
	// Initialize agents with more specific instructions
	managerAgent := &swarmgo.Agent{
		Name: "ManagerAgent",
		Instructions: `You are a manager agent responsible for coordinating research and analysis tasks.
Your responsibilities:
1. Break down the research topic into specific subtasks
2. Delegate tasks to appropriate agents
3. Review and synthesize the final results
4. Route tasks using "route to [agent]" syntax

When delegating:
- Send research tasks to ResearchAgent
- Send analysis tasks to AnalysisAgent
- Review the final analysis before completion`,
		Model: "gpt-4",
	}

	researchAgent := &swarmgo.Agent{
		Name: "ResearchAgent",
		Instructions: `You are a research agent responsible for gathering information.
Your responsibilities:
1. Conduct thorough research on assigned topics
2. Focus on credible and recent information
3. Organize findings in a clear structure
4. Route your findings to AnalysisAgent using "route to AnalysisAgent"`,
		Model: "gpt-4",
	}

	analysisAgent := &swarmgo.Agent{
		Name: "AnalysisAgent",
		Instructions: `You are an analysis agent responsible for interpreting research data.
Your responsibilities:
1. Analyze research findings for key insights
2. Identify trends and patterns
3. Draw meaningful conclusions
4. Route final analysis to ManagerAgent using "route to ManagerAgent"`,
		Model: "gpt-4",
	}

	// Load environment variables
	if err := dotenv.Load(); err != nil {
		log.Printf("Warning: .env file not found: %v", err)
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}

	// Create workflow with hierarchical type
	workflow := swarmgo.NewWorkflow(apiKey, llm.OpenAI, swarmgo.HierarchicalWorkflow)

	// Set up cycle handling with user intervention
	workflow.SetCycleHandling(swarmgo.ContinueOnCycle)
	workflow.SetCycleCallback(func(from, to string) (bool, error) {
		fmt.Printf("\n\033[93mCycle detected: %s -> %s\033[0m\n", from, to)
		fmt.Print("Do you want to continue the cycle for further refinement? (y/n): ")
		var response string
		fmt.Scanln(&response)
		return strings.ToLower(response) == "y", nil
	})

	// Add agents to teams
	workflow.AddAgentToTeam(managerAgent, swarmgo.SupervisorTeam)
	workflow.AddAgentToTeam(researchAgent, swarmgo.ResearchTeam)
	workflow.AddAgentToTeam(analysisAgent, swarmgo.AnalysisTeam)

	// Set up agent connections
	workflow.ConnectAgents(managerAgent.Name, researchAgent.Name)
	workflow.ConnectAgents(researchAgent.Name, analysisAgent.Name)
	workflow.ConnectAgents(analysisAgent.Name, managerAgent.Name)

	// Define user request
	userRequest := "Please conduct research and analyze the topic: 'The impact of AI on modern industries.'"

	// Execute workflow
	fmt.Println("\n\033[96mStarting Workflow Execution\033[0m")
	fmt.Println("=====================================")

	result, err := workflow.Execute(managerAgent.Name, userRequest)
	if err != nil {
		log.Fatalf("Error executing workflow: %v", err)
	}

	// Print workflow summary
	fmt.Printf("\n\033[96mWorkflow Summary\033[0m\n")
	fmt.Printf("Total Duration: %v\n", result.EndTime.Sub(result.StartTime))
	fmt.Printf("Total Steps: %d\n", len(result.Steps))

	// Print each step result
	fmt.Println("\n\033[96mDetailed Step Results\033[0m")
	for _, step := range result.Steps {
		printStepResult(step)
	}

	// Example: Use step results in code
	fmt.Println("\n\033[96mExample: Using Step Results\033[0m")

	// Get research findings from ResearchAgent step
	for _, step := range result.Steps {
		if step.AgentName == "ResearchAgent" {
			fmt.Println("\nResearch Findings:")
			for _, msg := range step.Output {
				if msg.Role == llm.RoleAssistant {
					fmt.Printf("%s\n", msg.Content)
				}
			}
			break
		}
	}

	// Get final analysis from last AnalysisAgent step
	var lastAnalysis string
	for i := len(result.Steps) - 1; i >= 0; i-- {
		if result.Steps[i].AgentName == "AnalysisAgent" {
			for _, msg := range result.Steps[i].Output {
				if msg.Role == llm.RoleAssistant {
					lastAnalysis = msg.Content
					break
				}
			}
			break
		}
	}

	if lastAnalysis != "" {
		fmt.Println("\nFinal Analysis:")
		fmt.Printf("%s\n", lastAnalysis)
	}
}
