package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/joho/godotenv"
	swarmgo "github.com/pffreitas/swarmgo"
	"github.com/pffreitas/swarmgo/llm"
)

func main() {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Printf("Warning: Error loading .env file: %v", err)
	}

	// Get API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Create dynamic workflow creator
	creator := swarmgo.NewDynamicWorkflowCreator(apiKey, llm.OpenAI)

	// Register base agent templates with pre-defined functions and behaviors
	creator.RegisterBaseAgent("Researcher", createResearcherAgent())
	creator.RegisterBaseAgent("Writer", createWriterAgent())
	creator.RegisterBaseAgent("Analyst", createAnalystAgent())
	creator.RegisterBaseAgent("Coder", createCoderAgent())
	creator.RegisterBaseAgent("Supervisor", createSupervisorAgent())

	// User task that needs a workflow
	userTask := "Research the latest advancements in quantum computing and create a summary report that includes code examples and performance analysis."

	// Create a context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	fmt.Println("Creating dynamic workflow for task:", userTask)
	fmt.Println("--------------------------------------------------")

	// Create workflow specification
	spec, err := creator.CreateWorkflowFromTask(ctx, userTask)
	if err != nil {
		log.Fatalf("Error creating workflow specification: %v", err)
	}

	// Display the created workflow specification
	fmt.Printf("Workflow Plan:\n")
	fmt.Printf("Main Goal: %s\n", spec.MainGoal)
	fmt.Printf("Workflow Type: %s\n", spec.WorkflowType)
	fmt.Printf("Entry Point: %s\n\n", spec.EntryPoint)

	fmt.Printf("Agents:\n")
	for _, agent := range spec.Agents {
		fmt.Printf("  - %s: %s\n", agent.Name, agent.Role)
		fmt.Printf("    Connections: %v\n", agent.Connections)
	}

	fmt.Printf("\nData Flow:\n")
	for _, flow := range spec.DataFlow {
		fmt.Printf("  %s -> %s: %s\n", flow.From, flow.To, flow.Description)
	}

	// Build the workflow from the specification
	workflow, err := creator.BuildWorkflow(spec)
	if err != nil {
		log.Fatalf("Error building workflow: %v", err)
	}

	fmt.Println("\nExecuting workflow...")
	fmt.Println("--------------------------------------------------")

	// Execute the workflow
	result, err := workflow.Execute(spec.EntryPoint, userTask)
	if err != nil {
		log.Fatalf("Error executing workflow: %v", err)
	}

	// Print workflow results
	fmt.Printf("\nWorkflow Execution Results:\n")
	fmt.Printf("Total Duration: %v\n", result.EndTime.Sub(result.StartTime))
	fmt.Printf("Total Steps: %d\n", len(result.Steps))

	// Print final output
	if len(result.FinalOutput) > 0 {
		lastMsg := result.FinalOutput[len(result.FinalOutput)-1]
		fmt.Printf("\nFinal Output:\n%s\n", lastMsg.Content)
	}
}

// Helper functions to create agent templates

func createResearcherAgent() *swarmgo.Agent {
	return &swarmgo.Agent{
		Name: "Researcher",
		Instructions: `You are a research specialist who gathers and organizes information.
Your responsibilities:
1. Search for and collect relevant information on the assigned topic
2. Organize information in a structured format
3. Identify key points and trends
4. Provide references for all information
5. Focus on accuracy and comprehensiveness`,
		Model: "gpt-4o",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "search_web",
				Description: "Search the web for information on a given topic",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{
							"type":        "string",
							"description": "The search query",
						},
					},
					"required": []interface{}{"query"},
				},
				Function: func(args map[string]interface{}, ctx map[string]interface{}) swarmgo.Result {
					// In a real implementation, this would connect to a search API
					query := args["query"].(string)
					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Simulated search results for: %s", query),
					}
				},
			},
		},
	}
}

func createWriterAgent() *swarmgo.Agent {
	return &swarmgo.Agent{
		Name: "Writer",
		Instructions: `You are a writing specialist who creates clear, engaging content.
Your responsibilities:
1. Create well-structured documents based on provided information
2. Ensure clarity and readability
3. Adapt tone and style to the target audience
4. Organize content logically with appropriate headings and sections
5. Summarize complex information effectively`,
		Model: "gpt-4o",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "format_document",
				Description: "Format a document with proper structure",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"content": map[string]interface{}{
							"type":        "string",
							"description": "The content to format",
						},
						"format": map[string]interface{}{
							"type":        "string",
							"description": "The desired format (markdown, plaintext, etc.)",
						},
					},
					"required": []interface{}{"content"},
				},
				Function: func(args map[string]interface{}, ctx map[string]interface{}) swarmgo.Result {
					content := args["content"].(string)
					format := "markdown" // Default
					if f, ok := args["format"].(string); ok {
						format = f
					}
					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Formatted content '%s' in %s format", content, format),
					}
				},
			},
		},
	}
}

func createAnalystAgent() *swarmgo.Agent {
	return &swarmgo.Agent{
		Name: "Analyst",
		Instructions: `You are an analysis specialist who evaluates information and draws insights.
Your responsibilities:
1. Analyze data and information objectively
2. Identify patterns, trends, and insights
3. Evaluate strengths and weaknesses
4. Compare and contrast different perspectives
5. Provide actionable recommendations based on analysis`,
		Model: "gpt-4o",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "analyze_data",
				Description: "Analyze data to extract insights",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"data": map[string]interface{}{
							"type":        "string",
							"description": "The data to analyze",
						},
						"analysis_type": map[string]interface{}{
							"type":        "string",
							"description": "The type of analysis to perform",
						},
					},
					"required": []interface{}{"data"},
				},
				Function: func(args map[string]interface{}, ctx map[string]interface{}) swarmgo.Result {
					data := args["data"].(string)
					analysisType := "general"
					if t, ok := args["analysis_type"].(string); ok {
						analysisType = t
					}
					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Analyzed data '%s' using %s analysis", data, analysisType),
					}
				},
			},
		},
	}
}

func createCoderAgent() *swarmgo.Agent {
	return &swarmgo.Agent{
		Name: "Coder",
		Instructions: `You are a coding specialist who develops and explains code.
Your responsibilities:
1. Write clear, efficient code in the appropriate language
2. Explain code functionality and design decisions
3. Adapt code examples to specific use cases
4. Ensure code quality and readability
5. Provide comments and documentation within code`,
		Model: "gpt-4o",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "generate_code",
				Description: "Generate code based on requirements",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"requirements": map[string]interface{}{
							"type":        "string",
							"description": "The requirements for the code",
						},
						"language": map[string]interface{}{
							"type":        "string",
							"description": "The programming language to use",
						},
					},
					"required": []interface{}{"requirements", "language"},
				},
				Function: func(args map[string]interface{}, ctx map[string]interface{}) swarmgo.Result {
					lang := args["language"].(string)
					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Generated %s code", lang),
					}
				},
			},
		},
	}
}

func createSupervisorAgent() *swarmgo.Agent {
	return &swarmgo.Agent{
		Name: "Supervisor",
		Instructions: `You are a coordination specialist who oversees project workflow.
Your responsibilities:
1. Break down complex tasks into manageable subtasks
2. Assign tasks to appropriate specialized agents
3. Track progress and ensure quality standards
4. Integrate outputs from different agents
5. Ensure the final deliverable meets requirements`,
		Model: "gpt-4o",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "assign_task",
				Description: "Assign a task to a specialized agent",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"task": map[string]interface{}{
							"type":        "string",
							"description": "The task to assign",
						},
						"agent": map[string]interface{}{
							"type":        "string",
							"description": "The agent to assign the task to",
						},
					},
					"required": []interface{}{"task", "agent"},
				},
				Function: func(args map[string]interface{}, ctx map[string]interface{}) swarmgo.Result {
					task := args["task"].(string)
					agent := args["agent"].(string)
					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Task '%s' assigned to %s", task, agent),
					}
				},
			},
		},
	}
}
