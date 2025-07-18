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
	// Initialize supervisor agent with detailed instructions
	supervisorAgent := &swarmgo.Agent{
		Name: "SupervisorAgent",
		Instructions: `You are the SupervisorAgent responsible for overseeing and coordinating tasks.
Your responsibilities:
1. Break down complex tasks into smaller subtasks
2. Assign tasks to appropriate task agents
3. Monitor progress and provide guidance
4. Synthesize results from task agents
5. Make final decisions on completeness and quality

Task Assignment Protocol:
1. For data collection tasks:
   - Use "route to DataAgent"
   - Provide specific data points to collect
   - Set clear expectations for data format

2. For analysis tasks:
   - Use "route to AnalystAgent"
   - Specify analysis requirements
   - Request specific report sections

3. For reviewing results:
   - Review completeness and quality
   - If revisions needed, route back to appropriate agent with specific feedback
   - If satisfied, mark as complete with "FINAL_APPROVED:"

Communication Style:
- Be direct and professional
- Focus on task assignment and coordination
- No need to apologize for previous messages
- Each message should build on previous context

Remember: Only use these exact routing commands:
- "route to DataAgent"
- "route to AnalystAgent"
Never use any other routing commands. You are the supervisor so routing does not happen there so you have to decide only between these two agents.`,
		Model: "gpt-4",
	}

	dataAgent := &swarmgo.Agent{
		Name: "DataAgent",
		Instructions: `You are responsible for data collection and processing.
Your responsibilities:
1. Gather required data based on supervisor's instructions
2. Process and organize the data
3. Report results back using "route to SupervisorAgent"
4. Make revisions if requested

Always format your response with clear sections:
- Data Collection Methods
- Key Findings
- Data Organization
- Next Steps

Communication Style:
- Be direct and professional
- Build on previous context
- No need to apologize for previous messages
- Focus on data collection and organization

Remember: Always route back to SupervisorAgent using exactly:
"route to SupervisorAgent"`,
		Model: "gpt-4",
	}

	analystAgent := &swarmgo.Agent{
		Name: "AnalystAgent",
		Instructions: `You are responsible for analysis and reporting.
Your responsibilities:
1. Analyze processed data
2. Generate comprehensive reports
3. Report results back using "route to SupervisorAgent"
4. Make revisions if requested

Always format your reports with:
- Executive Summary
- Detailed Analysis
- Key Insights
- Recommendations

For final reports, use this exact format:
"FINAL_REPORT: [your report content]"

Communication Style:
- Be direct and professional
- Build on previous context
- No need to apologize for previous messages
- Focus on analysis and insights

Remember: Always route back to SupervisorAgent using exactly:
"route to SupervisorAgent"`,
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

	// Create workflow with supervisor type
	workflow := swarmgo.NewWorkflow(apiKey, llm.OpenAI, swarmgo.SupervisorWorkflow)

	// Set up cycle handling for revision requests
	workflow.SetCycleHandling(swarmgo.ContinueOnCycle)
	workflow.SetCycleCallback(func(from, to string) (bool, error) {
		fmt.Printf("\n\033[93mRevision cycle detected: %s -> %s\033[0m\n", from, to)
		fmt.Print("Continue with revision? (y/n): ")
		var response string
		fmt.Scanln(&response)
		return strings.ToLower(response) == "y", nil
	})

	// Add agents to teams
	workflow.AddAgentToTeam(supervisorAgent, swarmgo.SupervisorTeam)
	workflow.AddAgentToTeam(dataAgent, swarmgo.DocumentTeam)
	workflow.AddAgentToTeam(analystAgent, swarmgo.AnalysisTeam)

	// Set supervisor as team leader
	workflow.SetTeamLeader(supervisorAgent.Name, swarmgo.SupervisorTeam)

	// Connect agents in supervisor pattern
	workflow.ConnectAgents(supervisorAgent.Name, dataAgent.Name)
	workflow.ConnectAgents(supervisorAgent.Name, analystAgent.Name)
	workflow.ConnectAgents(dataAgent.Name, supervisorAgent.Name)
	workflow.ConnectAgents(analystAgent.Name, supervisorAgent.Name)

	// Define the project task
	projectTask := "Create a comprehensive report on recent market trends in the tech industry. " +
		"Focus on AI and machine learning developments in the last year. " +
		"Include key players, emerging technologies, and market size estimates."

	// Execute workflow
	fmt.Println("\n\033[96mStarting Supervisor Workflow\033[0m")
	fmt.Println("=====================================")

	result, err := workflow.Execute(supervisorAgent.Name, projectTask)
	if err != nil {
		log.Fatalf("Error executing workflow: %v", err)
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

	// Extract final report
	fmt.Println("\n\033[96mFinal Report\033[0m")
	var finalReport string
	for i := len(result.Steps) - 1; i >= 0; i-- {
		step := result.Steps[i]
		if step.AgentName == analystAgent.Name {
			for _, msg := range step.Output {
				if msg.Role == llm.RoleAssistant && strings.Contains(msg.Content, "FINAL_REPORT:") {
					finalReport = strings.TrimPrefix(msg.Content, "FINAL_REPORT:")
					break
				}
			}
			if finalReport != "" {
				break
			}
		}
	}

	if finalReport != "" {
		fmt.Println(finalReport)
	} else {
		fmt.Println("No final report found in workflow output")
	}

	// Keep the program running to maintain WebSocket connections
	fmt.Println("\nPress Enter to exit...")
	fmt.Scanln()
}
