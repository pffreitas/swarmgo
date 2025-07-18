package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"
	swarmgo "github.com/pffreitas/swarmgo"
	"github.com/pffreitas/swarmgo/llm"
)

// Simple task management workflow example

func main() {
	// Load environment variables
	godotenv.Load()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Create a graph for our simple task management workflow
	builder := swarmgo.NewGraphBuilder("Task Management Workflow", "Simple workflow to handle task creation and assignment")

	// Define agents for the workflow

	// Task Creator agent - Creates and describes tasks
	taskCreatorAgent := &swarmgo.Agent{
		Name: "TaskCreator",
		Instructions: `You are a task creation specialist. 
Your responsibilities:
1. Help users define clear, specific tasks
2. Ensure tasks have proper descriptions
3. Make sure task priorities are set
4. Suggest deadlines based on priority

Always be clear and concise in your responses.`,
		Model: "gpt-3.5-turbo",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "create_task",
				Description: "Create a new task",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"title": map[string]interface{}{
							"type":        "string",
							"description": "Task title",
						},
						"description": map[string]interface{}{
							"type":        "string",
							"description": "Task description",
						},
						"priority": map[string]interface{}{
							"type":        "string",
							"description": "Task priority (high, medium, low)",
						},
						"deadline": map[string]interface{}{
							"type":        "string",
							"description": "Task deadline (YYYY-MM-DD)",
						},
					},
					"required": []interface{}{"title", "description", "priority"},
				},
				Function: func(args map[string]interface{}, contextVars map[string]interface{}) swarmgo.Result {
					// Create a task object
					task := map[string]interface{}{
						"id":          fmt.Sprintf("TASK-%d", time.Now().Unix()),
						"title":       args["title"],
						"description": args["description"],
						"priority":    args["priority"],
						"status":      "created",
						"created_at":  time.Now().Format(time.RFC3339),
					}

					if deadline, ok := args["deadline"].(string); ok {
						task["deadline"] = deadline
					}

					// Store in context
					if contextVars != nil {
						contextVars["current_task"] = task
					}

					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Task created: %s (ID: %s)", args["title"], task["id"]),
					}
				},
			},
		},
	}

	// Task Assigner agent - Assigns tasks to people
	taskAssignerAgent := &swarmgo.Agent{
		Name: "TaskAssigner",
		Instructions: `You are a task assignment specialist.
Your responsibilities:
1. Match tasks to appropriate team members based on skills
2. Balance workload across the team
3. Consider deadlines and priorities
4. Update task status after assignment

Be efficient and fair in assignments.`,
		Model: "gpt-3.5-turbo",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "list_team_members",
				Description: "List available team members",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
				Function: func(args map[string]interface{}, contextVars map[string]interface{}) swarmgo.Result {
					// Simulated team members
					teamMembers := []map[string]interface{}{
						{
							"id":       "TM001",
							"name":     "Alice Smith",
							"skills":   []string{"frontend", "design", "javascript"},
							"workload": "medium",
						},
						{
							"id":       "TM002",
							"name":     "Bob Johnson",
							"skills":   []string{"backend", "database", "python"},
							"workload": "low",
						},
						{
							"id":       "TM003",
							"name":     "Carol Williams",
							"skills":   []string{"project management", "testing", "documentation"},
							"workload": "high",
						},
					}

					// Store in context
					if contextVars != nil {
						contextVars["team_members"] = teamMembers
					}

					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Retrieved %d team members", len(teamMembers)),
					}
				},
			},
			{
				Name:        "assign_task",
				Description: "Assign a task to a team member",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"task_id": map[string]interface{}{
							"type":        "string",
							"description": "Task ID",
						},
						"team_member_id": map[string]interface{}{
							"type":        "string",
							"description": "Team member ID",
						},
						"notes": map[string]interface{}{
							"type":        "string",
							"description": "Assignment notes",
						},
					},
					"required": []interface{}{"task_id", "team_member_id"},
				},
				Function: func(args map[string]interface{}, contextVars map[string]interface{}) swarmgo.Result {
					taskID := args["task_id"].(string)
					teamMemberID := args["team_member_id"].(string)
					notes := ""
					if notesArg, ok := args["notes"]; ok {
						notes = notesArg.(string)
					}

					// Get current task from context
					var taskTitle string
					if task, ok := contextVars["current_task"].(map[string]interface{}); ok {
						task["status"] = "assigned"
						task["assigned_to"] = teamMemberID
						task["assignment_notes"] = notes
						task["assigned_at"] = time.Now().Format(time.RFC3339)
						contextVars["current_task"] = task
						taskTitle = task["title"].(string)
					}

					// Get team member name
					teamMemberName := "Unknown"
					if teamMembers, ok := contextVars["team_members"].([]map[string]interface{}); ok {
						for _, member := range teamMembers {
							if member["id"] == teamMemberID {
								teamMemberName = member["name"].(string)
								break
							}
						}
					}

					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Task '%s' (ID: %s) assigned to %s", taskTitle, taskID, teamMemberName),
					}
				},
			},
		},
	}

	// Task Tracker agent - Tracks task status
	taskTrackerAgent := &swarmgo.Agent{
		Name: "TaskTracker",
		Instructions: `You are a task tracking specialist.
Your responsibilities:
1. Track the status of all tasks
2. Record updates and progress
3. Send notifications for approaching deadlines
4. Generate status reports

Be thorough and organized in your tracking.`,
		Model: "gpt-3.5-turbo",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "update_task_status",
				Description: "Update the status of a task",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"task_id": map[string]interface{}{
							"type":        "string",
							"description": "Task ID",
						},
						"status": map[string]interface{}{
							"type":        "string",
							"description": "New status (in_progress, blocked, completed)",
						},
						"notes": map[string]interface{}{
							"type":        "string",
							"description": "Status update notes",
						},
					},
					"required": []interface{}{"task_id", "status"},
				},
				Function: func(args map[string]interface{}, contextVars map[string]interface{}) swarmgo.Result {
					taskID := args["task_id"].(string)
					status := args["status"].(string)
					notes := ""
					if notesArg, ok := args["notes"]; ok {
						notes = notesArg.(string)
					}

					// Get current task from context
					var taskTitle string
					if task, ok := contextVars["current_task"].(map[string]interface{}); ok {
						prevStatus := task["status"]
						task["status"] = status
						task["status_notes"] = notes
						task["last_updated"] = time.Now().Format(time.RFC3339)
						contextVars["current_task"] = task
						taskTitle = task["title"].(string)

						// Record status change in history
						history := []map[string]interface{}{}
						if hist, ok := task["status_history"].([]map[string]interface{}); ok {
							history = hist
						}
						history = append(history, map[string]interface{}{
							"from":      prevStatus,
							"to":        status,
							"timestamp": task["last_updated"],
							"notes":     notes,
						})
						task["status_history"] = history
					}

					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Task '%s' (ID: %s) status updated to: %s", taskTitle, taskID, status),
					}
				},
			},
			{
				Name:        "generate_report",
				Description: "Generate a status report for a task",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"task_id": map[string]interface{}{
							"type":        "string",
							"description": "Task ID",
						},
					},
					"required": []interface{}{"task_id"},
				},
				Function: func(args map[string]interface{}, contextVars map[string]interface{}) swarmgo.Result {
					taskID := args["task_id"].(string)

					// Get current task from context
					var report string
					if task, ok := contextVars["current_task"].(map[string]interface{}); ok {
						title := task["title"].(string)
						description := task["description"].(string)
						status := task["status"].(string)
						priority := task["priority"].(string)
						createdAt := task["created_at"].(string)

						report = fmt.Sprintf("TASK REPORT\n===========\nID: %s\nTitle: %s\nDescription: %s\nStatus: %s\nPriority: %s\nCreated: %s\n",
							taskID, title, description, strings.ToUpper(status), strings.ToUpper(priority), createdAt)

						if assignedTo, ok := task["assigned_to"].(string); ok {
							report += fmt.Sprintf("Assigned To: %s\n", assignedTo)
						}

						if deadline, ok := task["deadline"].(string); ok {
							report += fmt.Sprintf("Deadline: %s\n", deadline)
						}

						contextVars["task_report"] = report
					}

					return swarmgo.Result{
						Success: true,
						Data:    report,
					}
				},
			},
		},
	}

	// Add all agent nodes to the graph
	builder.WithAgent("creator", "Task Creator", taskCreatorAgent)
	builder.WithAgent("assigner", "Task Assigner", taskAssignerAgent)
	builder.WithAgent("tracker", "Task Tracker", taskTrackerAgent)

	// Add tracker nodes to track state transitions
	builder.WithNode("creator_tracker", "Creator Tracker", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		newState := state.Clone()
		newState["last_node"] = "creator"
		return newState, nil
	})

	builder.WithNode("assigner_tracker", "Assigner Tracker", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		newState := state.Clone()
		newState["last_node"] = "assigner"
		return newState, nil
	})

	builder.WithNode("tracker_tracker", "Tracker Tracker", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		newState := state.Clone()
		newState["last_node"] = "tracker"
		return newState, nil
	})

	// Create a router node
	builder.WithNode("router", "Task Router", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		// Router modifies state to track visits
		newState := state.Clone()

		// Initialize node_visits if it doesn't exist
		var visits map[string]int
		if visitsRaw, exists := newState["node_visits"]; exists {
			if v, ok := visitsRaw.(map[string]int); ok {
				visits = v
			} else {
				visits = make(map[string]int)
			}
		} else {
			visits = make(map[string]int)
		}

		// Get last node from state
		lastNode, _ := newState.GetString("last_node")

		// Update the node visit counter for the last node
		if lastNode != "" {
			visits[lastNode]++
			fmt.Printf("Incrementing visit count for %s to %d\n", lastNode, visits[lastNode])
		}

		// Initialize router visits if needed
		if _, ok := visits["router"]; !ok {
			visits["router"] = 0
		}

		// Track router visits separately to detect loops within the router
		visits["router"]++

		// Safety check for router loops
		if visits["router"] > 3 {
			fmt.Println("Router loop detected, forcing exit")
			newState["force_exit"] = true
		}

		// Store updated visits back in state
		newState["node_visits"] = visits

		return newState, nil
	})

	// Create intake node to initialize the workflow
	builder.WithNode("intake", "Task Intake", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		// Initialize the state with task request
		newState := state.Clone()

		// Create initial conversation
		initialConversation := []llm.Message{
			{
				Role:    llm.RoleUser,
				Content: "I need to create a new task for redesigning our website homepage.",
			},
		}

		// Add the conversation to state
		newState[swarmgo.MessageKey] = initialConversation

		// Track workflow phases
		newState["workflow_phase"] = "creation"

		return newState, nil
	})

	// Create exit node
	builder.WithNode("exit", "Task Completion", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		newState := state.Clone()

		// Get messages
		messagesRaw, exists := state[swarmgo.MessageKey]
		if !exists || messagesRaw == nil {
			return newState, nil
		}

		var messages []llm.Message
		messagesData, _ := json.Marshal(messagesRaw)
		if err := json.Unmarshal(messagesData, &messages); err != nil {
			return newState, nil
		}

		// Add completion message
		messages = append(messages, llm.Message{
			Role:    llm.RoleAssistant,
			Content: "Your task has been created, assigned, and tracked successfully. The workflow is now complete.",
		})

		newState[swarmgo.MessageKey] = messages
		newState["workflow_complete"] = true

		return newState, nil
	})

	// Router condition function to determine next node
	// Fixed router condition function to determine next node
	// Final fixed router condition function that accounts for var_ prefix in keys
	routerCondition := func(state swarmgo.GraphState) (swarmgo.NodeID, error) {
		// Check for forced exit
		if forceExit, ok := state.GetBool("force_exit"); ok && forceExit {
			fmt.Println("Forced exit activated, ending workflow")
			return "exit", nil
		}

		// Get the workflow phase
		phase, _ := state.GetString("workflow_phase")

		// Debug output for state
		fmt.Println("Current state keys:")
		for k := range state {
			fmt.Printf("  - %s\n", k)
		}

		// IMPORTANT FIX: Check for both current_task and var_current_task
		task, hasTask := state["current_task"]
		if !hasTask {
			task, hasTask = state["var_current_task"] // Check for var_current_task as well
		}

		report, hasReport := state["task_report"]
		_ = report
		if !hasReport {
			report, hasReport = state["var_task_report"] // Check for var_task_report as well
		}

		// Print debug information about the task
		if hasTask {
			taskData, _ := json.Marshal(task)
			fmt.Printf("Task data: %s\n", string(taskData))
		}

		// Get visit tracking
		visits := make(map[string]int)
		if visitsRaw, exists := state["node_visits"]; exists {
			if v, ok := visitsRaw.(map[string]int); ok {
				visits = v
			}
		}

		fmt.Printf("Router condition: phase=%s, has_task=%v, has_report=%v\n",
			phase, hasTask, hasReport)

		// Check messages for evidence of task creation
		messagesRaw, ok := state[swarmgo.MessageKey]
		if ok && messagesRaw != nil {
			var messages []llm.Message
			messagesData, _ := json.Marshal(messagesRaw)
			if err := json.Unmarshal(messagesData, &messages); err == nil {
				// Look for function call evidence
				for _, msg := range messages {
					// Task creation evidence
					if msg.Role == llm.RoleFunction &&
						msg.Name == "create_task" &&
						strings.Contains(msg.Content, "Task created") {
						fmt.Println("Found task creation evidence in message history")

						if !hasTask && phase == "creation" {
							fmt.Println("Forcing transition to assignment phase")
							state["workflow_phase"] = "assignment"
							return "assigner", nil
						}
					}

					// Task assignment evidence
					if msg.Role == llm.RoleFunction &&
						msg.Name == "assign_task" &&
						strings.Contains(msg.Content, "assigned to") {
						fmt.Println("Found task assignment evidence in message history")

						if phase == "assignment" {
							fmt.Println("Forcing transition to tracking phase")
							state["workflow_phase"] = "tracking"
							return "tracker", nil
						}
					}

					// Task report evidence
					if msg.Role == llm.RoleFunction &&
						msg.Name == "generate_report" &&
						strings.Contains(msg.Content, "TASK REPORT") {
						fmt.Println("Found task report evidence in message history")

						if phase == "tracking" {
							fmt.Println("Report generated, completing workflow")
							return "exit", nil
						}
					}
				}
			}
		}

		// Route based on workflow phase
		switch phase {
		case "creation":
			if hasTask {
				fmt.Println("Task created, moving to assignment phase")
				state["workflow_phase"] = "assignment"
				return "assigner", nil
			}
			return "creator", nil

		case "assignment":
			// Check for evidence of assignment in task object
			if hasTask {
				taskMap, ok := task.(map[string]interface{})
				if ok {
					assignedTo, hasAssignment := taskMap["assigned_to"]
					if hasAssignment && assignedTo != nil && assignedTo.(string) != "" {
						fmt.Println("Task assigned, moving to tracking phase")
						state["workflow_phase"] = "tracking"
						return "tracker", nil
					}
				}
			}

			// Check if we've been in assignment phase too long
			if visits["assigner"] >= 2 {
				fmt.Println("Assignment phase taking too long, forcing progression to tracking")
				state["workflow_phase"] = "tracking"
				return "tracker", nil
			}

			return "assigner", nil

		case "tracking":
			if hasReport {
				fmt.Println("Report generated, completing workflow")
				return "exit", nil
			}

			// If stuck in tracking phase too long, force exit
			if visits["tracker"] >= 2 {
				fmt.Println("Tracking phase taking too long, forcing completion")
				return "exit", nil
			}

			return "tracker", nil
		}

		// If phase is unknown or not set, start with creation
		state["workflow_phase"] = "creation"
		return "creator", nil
	}

	// Connect everything
	builder.WithEdge("intake", "creator")
	builder.WithEdge("creator", "creator_tracker")
	builder.WithEdge("creator_tracker", "router")

	builder.WithEdge("assigner", "assigner_tracker")
	builder.WithEdge("assigner_tracker", "router")

	builder.WithEdge("tracker", "tracker_tracker")
	builder.WithEdge("tracker_tracker", "router")

	// Add conditional edges from router to agents
	builder.WithConditionalEdge("router", "creator", routerCondition)
	builder.WithConditionalEdge("router", "assigner", routerCondition)
	builder.WithConditionalEdge("router", "tracker", routerCondition)
	builder.WithConditionalEdge("router", "exit", routerCondition)

	// Set entry and exit points
	builder.WithEntryPoint("intake")
	builder.WithExitPoint("exit")

	// Build the graph
	graph := builder.Build()

	// Create runner
	runner := swarmgo.NewGraphRunner()
	runner.RegisterGraph(graph)

	// Initialize state
	initialState := swarmgo.GraphState{
		"api_key":  apiKey,
		"provider": string(llm.OpenAI),
	}

	// Execute the workflow
	fmt.Println("Starting Task Management Workflow simulation...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	finalState, err := runner.ExecuteGraph(ctx, graph.ID, initialState)
	if err != nil {
		log.Fatalf("Error executing workflow: %v", err)
	}

	// Print final results
	fmt.Println("\nWorkflow completed successfully!")
	fmt.Println("Final state contains:")

	// Print conversation history
	if messagesRaw, ok := finalState[swarmgo.MessageKey]; ok {
		fmt.Println("\nConversation history:")
		var messages []llm.Message
		messagesData, _ := json.Marshal(messagesRaw)
		json.Unmarshal(messagesData, &messages)

		for i, msg := range messages {
			switch msg.Role {
			case llm.RoleUser:
				fmt.Printf("%d. User: %s\n", i+1, msg.Content)
			case llm.RoleAssistant:
				fmt.Printf("%d. Assistant: %s\n", i+1, msg.Content)
			case llm.RoleFunction:
				fmt.Printf("%d. System: [%s] %s\n", i+1, msg.Name, msg.Content)
			}
		}
	}

	// Print task information
	if taskRaw, ok := finalState["current_task"]; ok {
		fmt.Println("\nFinal Task:")
		taskData, _ := json.MarshalIndent(taskRaw, "", "  ")
		fmt.Println(string(taskData))
	}

	// Print task report if available
	if report, ok := finalState.GetString("task_report"); ok {
		fmt.Println("\nTask Report:")
		fmt.Println(report)
	}
}
