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

// Medical clinic workflow example using the new LangGraph-inspired workflow system

// FetchPatientRecord simulates retrieving patient records from a database
func FetchPatientRecord(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	patientID, ok := args["patient_id"].(string)
	if !ok {
		return swarmgo.Result{
			Success: false,
			Data:    "Error: patient_id is required",
		}
	}

	// Simulate database lookup
	patientData := map[string]interface{}{
		"id":        patientID,
		"name":      "John Doe",
		"age":       45,
		"gender":    "Male",
		"allergies": []string{"Penicillin"},
		"history": []map[string]interface{}{
			{
				"date":      "2023-10-15",
				"diagnosis": "Hypertension",
				"treatment": "Prescribed Lisinopril 10mg",
			},
			{
				"date":      "2024-01-22",
				"diagnosis": "Seasonal Allergies",
				"treatment": "Prescribed Cetirizine 10mg",
			},
		},
	}

	// Store in context
	if contextVariables != nil {
		contextVariables["patient"] = patientData
	}

	return swarmgo.Result{
		Success: true,
		Data:    fmt.Sprintf("Retrieved patient record for %s (John Doe, 45, Male)", patientID),
	}
}

// ScheduleAppointment simulates scheduling a patient appointment
func ScheduleAppointment(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	patientID, ok1 := args["patient_id"].(string)
	doctor, ok2 := args["doctor"].(string)
	date, ok3 := args["date"].(string)
	time, ok4 := args["time"].(string)

	if !ok1 || !ok2 || !ok3 || !ok4 {
		return swarmgo.Result{
			Success: false,
			Data:    "Error: patient_id, doctor, date and time are all required",
		}
	}

	appointment := map[string]interface{}{
		"patient_id": patientID,
		"doctor":     doctor,
		"date":       date,
		"time":       time,
		"status":     "scheduled",
	}

	if contextVariables != nil {
		contextVariables["appointment"] = appointment
	}

	return swarmgo.Result{
		Success: true,
		Data:    fmt.Sprintf("Appointment scheduled with Dr. %s on %s at %s", doctor, date, time),
	}
}

// OrderLabTest simulates ordering lab tests
func OrderLabTest(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	patientID, ok1 := args["patient_id"].(string)
	testType, ok2 := args["test_type"].(string)

	if !ok1 || !ok2 {
		return swarmgo.Result{
			Success: false,
			Data:    "Error: patient_id and test_type are required",
		}
	}

	labOrder := map[string]interface{}{
		"patient_id": patientID,
		"test_type":  testType,
		"status":     "ordered",
		"order_date": time.Now().Format("2006-01-02"),
	}

	if contextVariables != nil {
		contextVariables["lab_order"] = labOrder
	}

	return swarmgo.Result{
		Success: true,
		Data:    fmt.Sprintf("Lab test '%s' ordered for patient %s", testType, patientID),
	}
}

// PrescribeMedication simulates prescribing medication
func PrescribeMedication(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	patientID, ok1 := args["patient_id"].(string)
	medication, ok2 := args["medication"].(string)
	dosage, ok3 := args["dosage"].(string)
	frequency, ok4 := args["frequency"].(string)

	if !ok1 || !ok2 || !ok3 || !ok4 {
		return swarmgo.Result{
			Success: false,
			Data:    "Error: patient_id, medication, dosage and frequency are required",
		}
	}

	// Check for allergies
	if patientData, ok := contextVariables["patient"].(map[string]interface{}); ok {
		if allergies, ok := patientData["allergies"].([]string); ok {
			for _, allergy := range allergies {
				if strings.Contains(strings.ToLower(medication), strings.ToLower(allergy)) {
					return swarmgo.Result{
						Success: false,
						Data:    fmt.Sprintf("WARNING: Patient is allergic to %s", allergy),
					}
				}
			}
		}
	}

	prescription := map[string]interface{}{
		"patient_id": patientID,
		"medication": medication,
		"dosage":     dosage,
		"frequency":  frequency,
		"date":       time.Now().Format("2006-01-02"),
	}

	if contextVariables != nil {
		contextVariables["prescription"] = prescription
	}

	return swarmgo.Result{
		Success: true,
		Data:    fmt.Sprintf("Prescribed %s %s %s for patient %s", medication, dosage, frequency, patientID),
	}
}

func main() {
	// Load environment variables
	godotenv.Load()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Create a graph for our medical clinic workflow
	builder := swarmgo.NewGraphBuilder("Medical Clinic Workflow", "Workflow for patient processing in a medical clinic")

	// 1. Receptionist Agent - Handles patient intake and scheduling
	receptionistAgent := &swarmgo.Agent{
		Name: "Receptionist",
		Instructions: `You are a medical clinic receptionist. 
Your responsibilities:
1. Greet patients and collect their basic information
2. Verify patient records in the system
3. Schedule appointments with appropriate doctors
4. Direct patients to the right department

Always be professional, courteous, and efficient. Collect patient ID when possible.`,
		Model: "gpt-4",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "fetch_patient_record",
				Description: "Retrieve a patient's medical record",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"patient_id": map[string]interface{}{
							"type":        "string",
							"description": "Patient identification number",
						},
					},
					"required": []interface{}{"patient_id"},
				},
				Function: FetchPatientRecord,
			},
			{
				Name:        "schedule_appointment",
				Description: "Schedule an appointment for a patient",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"patient_id": map[string]interface{}{
							"type":        "string",
							"description": "Patient identification number",
						},
						"doctor": map[string]interface{}{
							"type":        "string",
							"description": "Doctor's name",
						},
						"date": map[string]interface{}{
							"type":        "string",
							"description": "Appointment date (YYYY-MM-DD)",
						},
						"time": map[string]interface{}{
							"type":        "string",
							"description": "Appointment time (HH:MM)",
						},
					},
					"required": []interface{}{"patient_id", "doctor", "date", "time"},
				},
				Function: ScheduleAppointment,
			},
		},
	}

	// 2. Nurse Agent - Takes vitals and prepares patients
	nurseAgent := &swarmgo.Agent{
		Name: "Nurse",
		Instructions: `You are a clinic nurse.
Your responsibilities:
1. Take patient vitals (blood pressure, temperature, etc.)
2. Record patient symptoms and concerns
3. Prepare patients for examination
4. Assist doctors during procedures

Be caring, attentive, and thorough in your assessments.`,
		Model: "gpt-4",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "fetch_patient_record",
				Description: "Retrieve a patient's medical record",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"patient_id": map[string]interface{}{
							"type":        "string",
							"description": "Patient identification number",
						},
					},
					"required": []interface{}{"patient_id"},
				},
				Function: FetchPatientRecord,
			},
			{
				Name:        "record_vitals",
				Description: "Record patient vitals",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"patient_id": map[string]interface{}{
							"type":        "string",
							"description": "Patient identification number",
						},
						"temperature": map[string]interface{}{
							"type":        "string",
							"description": "Body temperature in F or C",
						},
						"blood_pressure": map[string]interface{}{
							"type":        "string",
							"description": "Blood pressure reading (systolic/diastolic)",
						},
						"heart_rate": map[string]interface{}{
							"type":        "string",
							"description": "Heart rate in BPM",
						},
					},
					"required": []interface{}{"patient_id"},
				},
				Function: func(args map[string]interface{}, contextVars map[string]interface{}) swarmgo.Result {
					// Simple function to record vitals
					patientID := args["patient_id"].(string)

					vitals := map[string]interface{}{}
					for key, value := range args {
						if key != "patient_id" {
							vitals[key] = value
						}
					}

					if contextVars != nil {
						contextVars["vitals"] = vitals
					}

					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Recorded vitals for patient %s", patientID),
					}
				},
			},
		},
	}

	// 3. Doctor Agent - Diagnoses patients and prescribes treatment
	doctorAgent := &swarmgo.Agent{
		Name: "Doctor",
		Instructions: `You are a medical doctor at a clinic.
Your responsibilities:
1. Review patient history and current symptoms
2. Perform examinations and make diagnoses
3. Order appropriate tests and interpret results
4. Prescribe medications and treatments
5. Provide medical advice and follow-up plans

Be thorough, accurate, and compassionate in your care.`,
		Model: "gpt-4",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "fetch_patient_record",
				Description: "Retrieve a patient's medical record",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"patient_id": map[string]interface{}{
							"type":        "string",
							"description": "Patient identification number",
						},
					},
					"required": []interface{}{"patient_id"},
				},
				Function: FetchPatientRecord,
			},
			{
				Name:        "order_lab_test",
				Description: "Order laboratory tests for a patient",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"patient_id": map[string]interface{}{
							"type":        "string",
							"description": "Patient identification number",
						},
						"test_type": map[string]interface{}{
							"type":        "string",
							"description": "Type of test to order (e.g., blood panel, urinalysis)",
						},
					},
					"required": []interface{}{"patient_id", "test_type"},
				},
				Function: OrderLabTest,
			},
			{
				Name:        "prescribe_medication",
				Description: "Prescribe medication for a patient",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"patient_id": map[string]interface{}{
							"type":        "string",
							"description": "Patient identification number",
						},
						"medication": map[string]interface{}{
							"type":        "string",
							"description": "Medication name",
						},
						"dosage": map[string]interface{}{
							"type":        "string",
							"description": "Dosage amount",
						},
						"frequency": map[string]interface{}{
							"type":        "string",
							"description": "How often to take the medication",
						},
					},
					"required": []interface{}{"patient_id", "medication", "dosage", "frequency"},
				},
				Function: PrescribeMedication,
			},
		},
	}

	// 4. Lab Technician Agent - Processes lab tests
	labTechAgent := &swarmgo.Agent{
		Name: "LabTechnician",
		Instructions: `You are a medical laboratory technician.
Your responsibilities:
1. Process lab test orders
2. Collect specimens when necessary
3. Run laboratory tests
4. Record and report test results
5. Maintain lab equipment and standards

Be precise, methodical, and attentive to detail.`,
		Model: "gpt-3.5-turbo",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "fetch_patient_record",
				Description: "Retrieve a patient's medical record",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"patient_id": map[string]interface{}{
							"type":        "string",
							"description": "Patient identification number",
						},
					},
					"required": []interface{}{"patient_id"},
				},
				Function: FetchPatientRecord,
			},
			{
				Name:        "process_lab_test",
				Description: "Process a laboratory test",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"patient_id": map[string]interface{}{
							"type":        "string",
							"description": "Patient identification number",
						},
						"test_type": map[string]interface{}{
							"type":        "string",
							"description": "Type of test",
						},
						"results": map[string]interface{}{
							"type":        "string",
							"description": "Test results",
						},
					},
					"required": []interface{}{"patient_id", "test_type", "results"},
				},
				Function: func(args map[string]interface{}, contextVars map[string]interface{}) swarmgo.Result {
					// Process lab test and record results
					patientID := args["patient_id"].(string)
					testType := args["test_type"].(string)
					results := args["results"].(string)

					testResults := map[string]interface{}{
						"patient_id": patientID,
						"test_type":  testType,
						"results":    results,
						"date":       time.Now().Format("2006-01-02"),
						"status":     "completed",
					}

					if contextVars != nil {
						contextVars["test_results"] = testResults
					}

					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Processed %s test for patient %s with results: %s", testType, patientID, results),
					}
				},
			},
		},
	}

	// 5. Pharmacist Agent - Dispenses medications and provides instructions
	pharmacistAgent := &swarmgo.Agent{
		Name: "Pharmacist",
		Instructions: `You are a clinic pharmacist.
Your responsibilities:
1. Review medication orders for accuracy
2. Check for drug interactions and contradictions
3. Prepare and dispense medications
4. Provide medication information to patients
5. Ensure proper medication management

Be meticulous, knowledgeable, and patient-focused.`,
		Model: "gpt-3.5-turbo",
		Functions: []swarmgo.AgentFunction{
			{
				Name:        "fetch_patient_record",
				Description: "Retrieve a patient's medical record",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"patient_id": map[string]interface{}{
							"type":        "string",
							"description": "Patient identification number",
						},
					},
					"required": []interface{}{"patient_id"},
				},
				Function: FetchPatientRecord,
			},
			{
				Name:        "dispense_medication",
				Description: "Dispense medication for a patient",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"patient_id": map[string]interface{}{
							"type":        "string",
							"description": "Patient identification number",
						},
						"medication": map[string]interface{}{
							"type":        "string",
							"description": "Medication name",
						},
						"instructions": map[string]interface{}{
							"type":        "string",
							"description": "Instructions for taking the medication",
						},
					},
					"required": []interface{}{"patient_id", "medication", "instructions"},
				},
				Function: func(args map[string]interface{}, contextVars map[string]interface{}) swarmgo.Result {
					// Dispense medication
					patientID := args["patient_id"].(string)
					medication := args["medication"].(string)
					instructions := args["instructions"].(string)

					dispensed := map[string]interface{}{
						"patient_id":   patientID,
						"medication":   medication,
						"instructions": instructions,
						"date":         time.Now().Format("2006-01-02"),
						"status":       "dispensed",
					}

					if contextVars != nil {
						contextVars["dispensed_medication"] = dispensed
					}

					return swarmgo.Result{
						Success: true,
						Data:    fmt.Sprintf("Dispensed %s for patient %s with instructions: %s", medication, patientID, instructions),
					}
				},
			},
		},
	}

	// Add all agent nodes to the graph
	builder.WithAgent("reception", "Receptionist", receptionistAgent)
	builder.WithAgent("nurse", "Nurse", nurseAgent)
	builder.WithAgent("doctor", "Doctor", doctorAgent)
	builder.WithAgent("lab", "Lab Technician", labTechAgent)
	builder.WithAgent("pharmacy", "Pharmacist", pharmacistAgent)

	// Add tracker nodes to help with state transitions
	builder.WithNode("reception_tracker", "Reception Tracker", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		newState := state.Clone()
		newState["last_node"] = "reception"
		return newState, nil
	})

	builder.WithNode("nurse_tracker", "Nurse Tracker", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		newState := state.Clone()
		newState["last_node"] = "nurse"
		return newState, nil
	})

	builder.WithNode("doctor_tracker", "Doctor Tracker", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		newState := state.Clone()
		newState["last_node"] = "doctor"
		return newState, nil
	})

	builder.WithNode("lab_tracker", "Lab Tracker", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		newState := state.Clone()
		newState["last_node"] = "lab"
		return newState, nil
	})

	// Create a router node to determine where patient should go next
	builder.WithNode("router", "Patient Router", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
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

		// Get last node from state, which should be stored by the previous node
		lastNode, _ := newState.GetString("last_node")

		// Important: Update the node visit counter for the last node
		if lastNode != "" {
			visits[lastNode]++
			fmt.Printf("Incrementing visit count for %s to %d\n", lastNode, visits[lastNode])
		}

		// Initialize router visits if needed
		if _, ok := visits["router"]; !ok {
			visits["router"] = 0
		}

		// Important: Track router visits separately to detect loops within the router
		visits["router"]++

		// Safety check for router loops
		if visits["router"] > 3 {
			fmt.Println("Router loop detected, forcing exit")
			// Force the next node to be exit to break the loop
			newState["force_exit"] = true
		}

		// Store updated visits back in state
		newState["node_visits"] = visits

		// Set router as the current node for debugging
		newState["current_node"] = "router"

		return newState, nil
	})

	// Create the condition function for routing
	routerCondition := func(state swarmgo.GraphState) (swarmgo.NodeID, error) {
		// Check for forced exit first
		if forceExit, ok := state.GetBool("force_exit"); ok && forceExit {
			fmt.Println("Forced exit activated, ending workflow")
			return "exit", nil
		}

		// Init visit tracking if needed
		visits := make(map[string]int)
		if visitsRaw, exists := state["node_visits"]; exists {
			if v, ok := visitsRaw.(map[string]int); ok {
				visits = v
			}
		}

		// Get messages
		messagesRaw, ok := state[swarmgo.MessageKey]
		if !ok || messagesRaw == nil {
			return "reception", nil
		}

		var messages []llm.Message
		messagesData, _ := json.Marshal(messagesRaw)
		if err := json.Unmarshal(messagesData, &messages); err != nil {
			return "reception", nil
		}

		if len(messages) == 0 {
			return "reception", nil
		}

		// Print debugging info
		fmt.Printf("Router check: reception visits=%d, messages=%d, router visits=%d\n",
			visits["reception"], len(messages), visits["router"])

		// Safety valve: if we have too many messages or reception visits, force progression
		if len(messages) > 15 || visits["reception"] > 5 {
			fmt.Println("Force progression: too many messages or reception visits")

			// Force patient data into state if not already there
			if _, hasPatient := state["patient"]; !hasPatient {
				// Create simulated patient record
				patientID := "123456"
				patientData := map[string]interface{}{
					"id":        patientID,
					"name":      "John Doe",
					"age":       45,
					"gender":    "Male",
					"allergies": []string{"Penicillin"},
					"history": []map[string]interface{}{
						{
							"date":      "2023-10-15",
							"diagnosis": "Hypertension",
							"treatment": "Prescribed Lisinopril 10mg",
						},
						{
							"date":      "2024-01-22",
							"diagnosis": "Seasonal Allergies",
							"treatment": "Prescribed Cetirizine 10mg",
						},
					},
				}
				state["patient"] = patientData

				// Add function call result to messages
				if messagesArr, ok := messagesRaw.([]llm.Message); ok {
					messagesArr = append(messagesArr, llm.Message{
						Role:    llm.RoleFunction,
						Name:    "fetch_patient_record",
						Content: fmt.Sprintf("Retrieved patient record for %s (John Doe, 45, Male)", patientID),
					})
					state[swarmgo.MessageKey] = messagesArr
				}
			}

			// Force visit to doctor with all data populated
			// Add simulated vitals to force progression
			if _, hasVitals := state["vitals"]; !hasVitals {
				state["vitals"] = map[string]interface{}{
					"temperature":    "98.6 F",
					"blood_pressure": "120/80",
					"heart_rate":     "72 bpm",
				}
			}

			return "exit", nil // Directly go to exit to avoid further processing
		}

		// Check for exit keywords
		content := ""
		if len(messages) > 0 {
			content = strings.ToLower(messages[len(messages)-1].Content)
		}

		if strings.Contains(content, "checkout") ||
			strings.Contains(content, "done") ||
			strings.Contains(content, "complete") ||
			strings.Contains(content, "finished") ||
			strings.Contains(content, "thank you") {
			return "exit", nil
		}

		// Check if patient data exists
		_, hasPatientData := state["patient"]
		if !hasPatientData {
			// Route based on message content
			if strings.Contains(content, "test") || strings.Contains(content, "lab") {
				return "lab", nil
			} else if strings.Contains(content, "medication") || strings.Contains(content, "prescription") {
				return "pharmacy", nil
			} else {
				// Normal progression
				_, hasVitals := state["vitals"]
				if !hasVitals {
					return "nurse", nil
				} else {
					return "doctor", nil
				}
			}
		}

		// Check last message for patient record retrieval
		for i := len(messages) - 1; i >= 0; i-- {
			msg := messages[i]
			if msg.Role == llm.RoleFunction &&
				msg.Name == "fetch_patient_record" &&
				strings.Contains(msg.Content, "Retrieved patient record") {
				return "nurse", nil // Move to nurse after patient record retrieved
			}
		}

		// Default to staying at reception
		return "reception", nil
	}

	builder.WithNode("intake", "Patient Intake", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		// This node initializes the state with a more complete patient conversation
		newState := state.Clone()

		// Create initial conversation with patient response
		initialConversation := []llm.Message{
			{
				Role:    llm.RoleUser,
				Content: "Hello, I'd like to see a doctor today. My name is John Doe.",
			},
			// First agent response would go here in a real conversation
			{
				Role:    llm.RoleAssistant,
				Content: "Hello, Mr. Doe. Welcome to our clinic. Could you please provide me with your patient ID for verification?",
			},
			// Simulated patient response with ID
			{
				Role:    llm.RoleUser,
				Content: "Yes, my patient ID is JD1203.",
			},
			// Simulate function call to fetch patient record
			{
				Role:    llm.RoleFunction,
				Name:    "fetch_patient_record",
				Content: "Retrieved patient record for JD1203 (John Doe, 45, Male)",
			},
			// Add simulated agent acknowledgment
			{
				Role:    llm.RoleAssistant,
				Content: "Thank you, Mr. Doe. I've found your record. Can you tell me what brings you in today?",
			},
			// Add simulated patient reason for visit
			{
				Role:    llm.RoleUser,
				Content: "I've been having headaches and feeling dizzy for the past few days.",
			},
		}

		// Add the conversation to state
		newState[swarmgo.MessageKey] = initialConversation

		// Also add the patient data to state directly
		patientData := map[string]interface{}{
			"id":        "JD1203",
			"name":      "John Doe",
			"age":       45,
			"gender":    "Male",
			"allergies": []string{"Penicillin"},
			"history": []map[string]interface{}{
				{
					"date":      "2023-10-15",
					"diagnosis": "Hypertension",
					"treatment": "Prescribed Lisinopril 10mg",
				},
				{
					"date":      "2024-01-22",
					"diagnosis": "Seasonal Allergies",
					"treatment": "Prescribed Cetirizine 10mg",
				},
			},
		}
		newState["patient"] = patientData

		return newState, nil
	})

	builder.WithNode("exit", "Checkout Process", func(ctx context.Context, state swarmgo.GraphState) (swarmgo.GraphState, error) {
		// Process checkout, billing, etc.
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

		// Add checkout message
		messages = append(messages, llm.Message{
			Role:    llm.RoleAssistant,
			Content: "Thank you for visiting our clinic today. Your visit has been processed and any applicable charges have been sent to billing.",
		})

		newState[swarmgo.MessageKey] = messages
		newState["checkout_complete"] = true

		return newState, nil
	})

	// Set up the main workflow connections with trackers
	builder.WithEdge("intake", "reception")
	builder.WithEdge("reception", "reception_tracker")
	builder.WithEdge("reception_tracker", "router")

	builder.WithEdge("nurse", "nurse_tracker")
	builder.WithEdge("nurse_tracker", "router")

	builder.WithEdge("doctor", "doctor_tracker")
	builder.WithEdge("doctor_tracker", "router")

	builder.WithEdge("lab", "lab_tracker")
	builder.WithEdge("lab_tracker", "router")

	builder.WithEdge("pharmacy", "exit")

	// Add conditional edges from router to all possible nodes
	builder.WithConditionalEdge("router", "reception", routerCondition)
	builder.WithConditionalEdge("router", "nurse", routerCondition)
	builder.WithConditionalEdge("router", "doctor", routerCondition)
	builder.WithConditionalEdge("router", "lab", routerCondition)
	builder.WithConditionalEdge("router", "pharmacy", routerCondition)
	builder.WithConditionalEdge("router", "exit", routerCondition)

	// Set entry and exit points
	builder.WithEntryPoint("intake")
	builder.WithExitPoint("exit")

	// Build the complete graph
	graph := builder.Build()

	// Create a graph runner to execute the workflow
	runner := swarmgo.NewGraphRunner()
	runner.RegisterGraph(graph)

	// Initialize state with API key and model info
	initialState := swarmgo.GraphState{
		"api_key":  apiKey,
		"provider": string(llm.OpenAI),
	}

	// Execute the workflow (this would typically be triggered by an API endpoint or scheduler)
	fmt.Println("Starting Medical Clinic Workflow simulation...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	finalState, err := runner.ExecuteGraph(ctx, graph.ID, initialState)
	if err != nil {
		log.Fatalf("Error executing workflow: %v", err)
	}

	// Print final results
	fmt.Println("\nWorkflow completed successfully!")
	fmt.Println("Final state contains:")

	for key, value := range finalState {
		if key == swarmgo.MessageKey {
			fmt.Println("\nConversation history:")
			var messages []llm.Message
			messagesData, _ := json.Marshal(value)
			json.Unmarshal(messagesData, &messages)

			for i, msg := range messages {
				switch msg.Role {
				case llm.RoleUser:
					fmt.Printf("%d. Patient: %s\n", i+1, msg.Content)
				case llm.RoleAssistant:
					fmt.Printf("%d. Clinic Staff: %s\n", i+1, msg.Content)
				case llm.RoleFunction:
					fmt.Printf("%d. System: [%s] %s\n", i+1, msg.Name, msg.Content)
				}
			}
		} else if key == "checkout_complete" {
			fmt.Println("\nCheckout status: Complete")
		} else if strings.HasPrefix(string(key), "var_") {
			fmt.Printf("\nVariable %s: %v\n", strings.TrimPrefix(string(key), "var_"), value)
		} else if key == "patient" || key == "prescription" || key == "vitals" ||
			key == "test_results" || key == "dispensed_medication" {
			fmt.Printf("\n%s: %v\n", key, value)
		}
	}
}
