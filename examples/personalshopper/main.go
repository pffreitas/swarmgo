package main

import (
	"fmt"
	"log"
	"os"

	dotenv "github.com/joho/godotenv"
	swarm "github.com/pffreitas/swarmgo"
	"github.com/pffreitas/swarmgo/llm"
)

func main() {
	initializeDatabase()
	// Preview tables
	fmt.Println("Preview of Users table:")
	previewTable("Users")
	fmt.Println("\nPreview of PurchaseHistory table:")
	previewTable("PurchaseHistory")
	fmt.Println("\nPreview of Products table:")
	previewTable("Products")

	// Create agents
	refundsAgent := &swarm.Agent{
		Name: "RefundsAgent",
		Instructions: `You are a refund agent that handles all actions related to refunds after a return has been processed.
You must ask for both the user ID and item ID to initiate a refund. Ask for both user_id and item_id in one message.
If the user asks you to notify them, you must ask them what their preferred method of notification is. For notifications, you must
ask them for user_id and method in one message.`,
		Functions: []swarm.AgentFunction{
			{
				Name:        "refundItem",
				Description: "Initiate a refund based on the user ID and item ID.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"user_id": map[string]interface{}{
							"type":        "string",
							"description": "The user ID.",
						},
						"item_id": map[string]interface{}{
							"type":        "string",
							"description": "The item ID.",
						},
					},
					"required": []interface{}{"user_id", "item_id"},
				},
				Function: refundItem,
			},
			{
				Name:        "notifyCustomer",
				Description: "Notify a customer by their preferred method of either phone or email.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"user_id": map[string]interface{}{
							"type":        "string",
							"description": "The user ID.",
						},
						"method": map[string]interface{}{
							"type":        "string",
							"description": "The method of notification (email or phone).",
						},
					},
					"required": []interface{}{"user_id", "method"},
				},
				Function: notifyCustomer,
			},
		},
		Model: "gpt-4",
	}

	salesAgent := &swarm.Agent{
		Name: "SalesAgent",
		Instructions: `You are a sales agent that handles all actions related to placing an order to purchase an item.
Regardless of what the user wants to purchase, you must ask for BOTH the user ID and product ID to place an order.
An order cannot be placed without these two pieces of information. Ask for both user_id and product_id in one message.
If the user asks you to notify them, you must ask them what their preferred method is. For notifications, you must
ask them for user_id and method in one message.`,
		Functions: []swarm.AgentFunction{
			{
				Name:        "orderItem",
				Description: "Place an order for a product based on the user ID and product ID.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"user_id": map[string]interface{}{
							"type":        "string",
							"description": "The user ID.",
						},
						"product_id": map[string]interface{}{
							"type":        "string",
							"description": "The product ID.",
						},
					},
					"required": []interface{}{"user_id", "product_id"},
				},
				Function: orderItem,
			},
			{
				Name:        "notifyCustomer",
				Description: "Notify a customer by their preferred method of either phone or email.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"user_id": map[string]interface{}{
							"type":        "string",
							"description": "The user ID.",
						},
						"method": map[string]interface{}{
							"type":        "string",
							"description": "The method of notification (email or phone).",
						},
					},
					"required": []interface{}{"user_id", "method"},
				},
				Function: notifyCustomer,
			},
		},
		Model: "gpt-4",
	}

	// Define transfer functions for triage
	transferToSalesAgent := swarm.AgentFunction{
		Name:        "transferToSalesAgent",
		Description: "Transfer the conversation to the SalesAgent.",
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
		Function: func(args map[string]interface{}, contextVariables map[string]interface{}) swarm.Result {
			return swarm.Result{
				Agent: salesAgent,
				Data:  "Transferring to SalesAgent.",
			}
		},
	}

	transferToRefundsAgent := swarm.AgentFunction{
		Name:        "transferToRefundsAgent",
		Description: "Transfer the conversation to the RefundsAgent.",
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
		Function: func(args map[string]interface{}, contextVariables map[string]interface{}) swarm.Result {
			return swarm.Result{
				Agent: refundsAgent,
				Data:  "Transferring to RefundsAgent.",
			}
		},
	}

	triageAgent := &swarm.Agent{
		Name: "TriageAgent",
		Instructions: `You are to triage a user's request and call a tool to transfer to the right intent.
Once you are ready to transfer to the right intent, call the tool to transfer to the right intent.
You don't need to know specifics, just the topic of the request.
If the user's request is about making an order or purchasing an item, transfer to the SalesAgent.
If the user's request is about getting a refund on an item or returning a product, transfer to the RefundsAgent.
When you need more information to triage the request to an agent, ask a direct question without explaining why you're asking it.
Do not share your thought process with the user! Do not make unreasonable assumptions on behalf of the user.`,
		Functions: []swarm.AgentFunction{
			transferToSalesAgent,
			transferToRefundsAgent,
		},
		Model: "gpt-4",
	}
	dotenv.Load()
	// Initialize Swarm client
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("Please set the OPENAI_API_KEY environment variable.")
	}

	client := swarm.NewSwarm(apiKey, llm.OpenAI)

	swarm.RunDemoLoop(client, triageAgent)
}
