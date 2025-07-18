package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	dotenv "github.com/joho/godotenv"
	swarmgo "github.com/pffreitas/swarmgo"
	"github.com/pffreitas/swarmgo/llm"
)

// getWeather simulates getting weather data for a location
func getWeather(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	location := args["location"].(string)
	time := "now"
	if t, ok := args["time"].(string); ok {
		time = t
	}

	// Store in context variables (very important)
	if contextVariables != nil {
		contextVariables["last_weather_location"] = location
		contextVariables["last_weather_temp"] = 65
		contextVariables["has_weather_data"] = true
	}

	return swarmgo.Result{
		Success: true,
		Data:    fmt.Sprintf("The temperature in %s is 65 degrees at %s.", location, time),
	}
}

// sendEmail simulates sending an email
func sendEmail(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	recipient := args["recipient"].(string)
	subject := args["subject"].(string)
	body := args["body"].(string)

	fmt.Println("Sending email...")
	fmt.Printf("To: %s\nSubject: %s\nBody: %s\n", recipient, subject, body)

	// Store in context variables (very important)
	if contextVariables != nil {
		contextVariables["last_email_recipient"] = recipient
		contextVariables["last_email_subject"] = subject
		contextVariables["email_sent"] = true
	}

	return swarmgo.Result{
		Success: true,
		Data:    fmt.Sprintf("Email successfully sent to %s with subject '%s'", recipient, subject),
	}
}

// Main function - the key improvements are in how we set up the agent
func main() {
	dotenv.Load()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}

	// Create new client
	client := swarmgo.NewSwarm(apiKey, llm.OpenAI)

	// Create agent with VERY SPECIFIC instructions
	weatherAgent := &swarmgo.Agent{
		Name: "WeatherAgent",
		Instructions: `You are a helpful weather assistant with the ability to check weather data and send emails.

IMPORTANT CAPABILITIES:
1. You CAN get real-time weather data using the getWeather function
2. You CAN send emails using the sendEmail function

When you use these functions, you must:
- Acknowledge the successful function execution in your response
- Reference the specific data returned by the function
- NEVER say you cannot perform these functions - you absolutely can!

For emails:
- When the sendEmail function returns "Email sent successfully" or similar, this means the email was ACTUALLY SENT
- Confirm to the user that you have sent the email
- Reference the recipient in your confirmation

Remember, you have FULL ACCESS to both weather data and email functionality. Use them confidently.`,

		Functions: []swarmgo.AgentFunction{
			{
				Name:        "getWeather",
				Description: "Get the current weather in a given location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "The city to get the weather for",
						},
						"time": map[string]interface{}{
							"type":        "string",
							"description": "The time to get the weather for",
						},
					},
					"required": []interface{}{"location"},
				},
				Function: getWeather,
			},
			{
				Name:        "sendEmail",
				Description: "Send an email to a recipient",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"recipient": map[string]interface{}{
							"type":        "string",
							"description": "The recipient of the email",
						},
						"subject": map[string]interface{}{
							"type":        "string",
							"description": "The subject of the email",
						},
						"body": map[string]interface{}{
							"type":        "string",
							"description": "The body of the email",
						},
					},
					"required": []interface{}{"recipient", "subject", "body"},
				},
				Function: sendEmail,
			},
		},
		Model: "gpt-4",
		// Add instructions function to use context variables
		InstructionsFunc: func(contextVariables map[string]interface{}) string {
			baseInstructions := `You are a helpful weather assistant with the ability to check weather data and send emails.

IMPORTANT CAPABILITIES:
1. You CAN get real-time weather data using the getWeather function
2. You CAN send emails using the sendEmail function

When you use these functions, you must:
- Acknowledge the successful function execution in your response
- Reference the specific data returned by the function
- NEVER say you cannot perform these functions - you absolutely can!

For emails:
- When the sendEmail function returns "Email sent successfully" or similar, this means the email was ACTUALLY SENT
- Confirm to the user that you have sent the email
- Reference the recipient in your confirmation

Remember, you have FULL ACCESS to both weather data and email functionality. Use them confidently.`

			// Add context-specific instructions
			if emailSent, ok := contextVariables["email_sent"].(bool); ok && emailSent {
				recipient, _ := contextVariables["last_email_recipient"].(string)
				baseInstructions += fmt.Sprintf("\n\nRECENT ACTION: You have just sent an email to %s. Make sure to confirm this in your response.", recipient)
			}

			if hasWeather, ok := contextVariables["has_weather_data"].(bool); ok && hasWeather {
				location, _ := contextVariables["last_weather_location"].(string)
				temp, _ := contextVariables["last_weather_temp"].(float64)
				baseInstructions += fmt.Sprintf("\n\nRECENT ACTION: You have just checked the weather in %s and found it to be %.0f degrees. Use this in your response if relevant.", location, temp)
			}

			return baseInstructions
		},
	}

	// Run customized demo loop with debug information
	fmt.Println("==== Starting SwarmGo CLI Demo ====")
	fmt.Printf("Agent: %s\nModel: %s\n\n", weatherAgent.Name, weatherAgent.Model)

	// Context variables to track state across turns
	contextVariables := make(map[string]interface{})
	messages := []llm.Message{}
	reader := bufio.NewReader(os.Stdin)

	for {
		// Get user input
		fmt.Print("You: ")
		userInput, _ := reader.ReadString('\n')
		userInput = strings.TrimSpace(userInput)

		if userInput == "exit" || userInput == "quit" || userInput == "q" {
			break
		}

		// Add to messages
		messages = append(messages, llm.Message{
			Role:    llm.RoleUser,
			Content: userInput,
		})

		// Create context with timeout
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

		// Show thinking indicator
		fmt.Print("Thinking...")

		// Execute agent with our context variables
		response, err := client.Run(ctx, weatherAgent, messages, contextVariables, "", false, false, 1, true)

		// Clear indicator
		fmt.Print("\r           \r")
		cancel()

		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		// Process response
		for _, msg := range response.Messages {
			switch msg.Role {
			case llm.RoleAssistant:
				if msg.Content != "" {
					fmt.Printf("%s: %s\n", weatherAgent.Name, msg.Content)
					messages = append(messages, msg)
				}
			case llm.RoleFunction:
				fmt.Printf("%s function result: %s\n", msg.Name, msg.Content)
				messages = append(messages, msg)
			}
		}
	}

	fmt.Println("Goodbye!")
}
