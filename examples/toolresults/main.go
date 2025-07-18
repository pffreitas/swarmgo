package main

import (
	"context"
	"fmt"
	"os"

	dotenv "github.com/joho/godotenv"
	swarmgo "github.com/pffreitas/swarmgo"
	"github.com/pffreitas/swarmgo/llm"
)

func calculateSum(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	num1 := int(args["num1"].(float64))
	num2 := int(args["num2"].(float64))
	sum := num1 + num2
	return swarmgo.Result{
		Success: true,
		Data:    fmt.Sprintf("The sum of %d and %d is %d", num1, num2, sum),
	}
}

func calculateProduct(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	num1 := int(args["num1"].(float64))
	num2 := int(args["num2"].(float64))
	product := num1 * num2
	return swarmgo.Result{
		Success: true,
		Data:    fmt.Sprintf("The product of %d and %d is %d", num1, num2, product),
	}
}

func main() {
	dotenv.Load()

	client := swarmgo.NewSwarm(os.Getenv("OPENAI_API_KEY"), llm.OpenAI)

	mathAgent := &swarmgo.Agent{
		Name:         "MathAgent",
		Instructions: "You are a math assistant. When given two numbers, calculate both their sum and product.",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "calculateSum",
				Description: "Calculate the sum of two numbers",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"num1": map[string]interface{}{
							"type":        "number",
							"description": "First number",
						},
						"num2": map[string]interface{}{
							"type":        "number",
							"description": "Second number",
						},
					},
					"required": []interface{}{"num1", "num2"},
				},
				Function: calculateSum,
			},
			{
				Name:        "calculateProduct",
				Description: "Calculate the product of two numbers",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"num1": map[string]interface{}{
							"type":        "number",
							"description": "First number",
						},
						"num2": map[string]interface{}{
							"type":        "number",
							"description": "Second number",
						},
					},
					"required": []interface{}{"num1", "num2"},
				},
				Function: calculateProduct,
			},
		},
		Model: "gpt-4",
	}

	// Create context
	ctx := context.Background()

	// Example message asking to perform calculations
	messages := []llm.Message{
		{
			Role:    llm.RoleUser,
			Content: "Calculate the sum and product of 5 and 3",
		},
	}

	// Run the agent with tool execution enabled
	response, err := client.Run(ctx, mathAgent, messages, nil, "", false, true, 1, true)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Print the final response from the agent
	fmt.Println("\nAgent's Final Response:")
	for _, msg := range response.Messages {
		if msg.Role == llm.RoleAssistant {
			fmt.Printf("Assistant: %s\n", msg.Content)
		}
	}

	// Print detailed information about tool calls
	fmt.Println("\nTool Call Results:")
	for _, result := range response.ToolResults {
		fmt.Printf("\nTool: %s\n", result.ToolName)
		fmt.Printf("Arguments: %v\n", result.Args)
		fmt.Printf("Result: %v\n", result.Result.Data)

		// You can also check if the tool call was successful
		if result.Result.Success {
			fmt.Printf("Status: Success\n")
		} else {
			fmt.Printf("Status: Failed\nError: %v\n", result.Result.Error)
		}
	}
}
