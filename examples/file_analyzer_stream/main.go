package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/pffreitas/swarmgo"
	"github.com/pffreitas/swarmgo/llm"
)

// CustomStreamHandler handles streaming responses with progress updates
type CustomStreamHandler struct {
	totalTokens int
	startTime   time.Time
}

func (h *CustomStreamHandler) OnStart() {
	h.startTime = time.Now()
	fmt.Println("🚀 Starting file analysis...")
}

func (h *CustomStreamHandler) OnToken(token string) {
	h.totalTokens++
	// Print progress every 50 tokens with tokens per second
	if h.totalTokens%50 == 0 {
		elapsed := time.Since(h.startTime).Seconds()
		tokensPerSec := float64(h.totalTokens) / elapsed
		fmt.Printf("\r📝 Processing... (%d tokens, %.1f tokens/sec)", h.totalTokens, tokensPerSec)
	}
	fmt.Print(token)
}

func (h *CustomStreamHandler) OnComplete(msg llm.Message) {
	elapsed := time.Since(h.startTime).Seconds()
	tokensPerSec := float64(h.totalTokens) / elapsed
	fmt.Printf("\n\n✅ Analysis complete! Processed %d tokens in %.1f seconds (%.1f tokens/sec)\n",
		h.totalTokens, elapsed, tokensPerSec)
}

func (h *CustomStreamHandler) OnError(err error) {
	fmt.Printf("❌ Error: %v\n", err)
}

func (h *CustomStreamHandler) OnToolCall(toolCall llm.ToolCall) {
	fmt.Printf("\n🔧 Using tool: %s\n", toolCall.Function.Name)
}

// FileProcessor handles file operations
type FileProcessor struct {
	maxFileSize int64 // maximum file size in bytes
}

func NewFileProcessor(maxFileSize int64) *FileProcessor {
	return &FileProcessor{
		maxFileSize: maxFileSize,
	}
}

func (fp *FileProcessor) ReadFile(path string) (string, error) {
	// Check if file exists
	fileInfo, err := os.Stat(path)
	if os.IsNotExist(err) {
		return "", fmt.Errorf("file does not exist: %s", path)
	}
	if err != nil {
		return "", fmt.Errorf("error accessing file: %v", err)
	}

	// Check file size
	if fileInfo.Size() > fp.maxFileSize {
		return "", fmt.Errorf("file size %d bytes exceeds maximum allowed size of %d bytes",
			fileInfo.Size(), fp.maxFileSize)
	}

	// Read file
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("error reading file: %v", err)
	}
	return string(content), nil
}

func (fp *FileProcessor) CountWords(text string) int {
	words := strings.Fields(text)
	return len(words)
}

func main() {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Printf("Warning: Error loading .env file: %v", err)
	}

	// Get OpenAI API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Get the project root directory from environment or use current directory
	projectRoot := os.Getenv("PROJECT_ROOT")
	if projectRoot == "" {
		var err error
		projectRoot, err = os.Getwd()
		if err != nil {
			log.Fatalf("Error getting current directory: %v", err)
		}
	}
	readmePath := filepath.Join(projectRoot, "Readme.md")

	// Create file processor with 10MB max file size
	fileProcessor := NewFileProcessor(10 * 1024 * 1024)

	// Verify the file exists and is readable
	if _, err := os.Stat(readmePath); err != nil {
		if os.IsNotExist(err) {
			log.Fatalf("README.md not found at %s", readmePath)
		}
		log.Fatalf("Error accessing README.md: %v", err)
	}

	// Define agent functions
	functions := []swarmgo.AgentFunction{
		{
			Name:        "read_file",
			Description: "Read content from a file",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the file to read",
					},
				},
				"required": []string{"path"},
			},
			Function: func(args map[string]interface{}, context map[string]interface{}) swarmgo.Result {
				path, ok := args["path"].(string)
				fmt.Println(path)
				if !ok {
					return swarmgo.Result{
						Error: fmt.Errorf("invalid path argument"),
					}
				}
				content, err := fileProcessor.ReadFile(path)
				if err != nil {
					return swarmgo.Result{
						Error: err,
						Data:  fmt.Sprintf("Error reading file: %v", err),
					}
				}
				return swarmgo.Result{
					Data: content,
				}
			},
		},
		{
			Name:        "count_words",
			Description: "Count words in a text",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"text": map[string]interface{}{
						"type":        "string",
						"description": "Text to analyze",
					},
				},
				"required": []string{"text"},
			},
			Function: func(args map[string]interface{}, context map[string]interface{}) swarmgo.Result {
				text, ok := args["text"].(string)
				if !ok {
					return swarmgo.Result{
						Error: fmt.Errorf("invalid text argument"),
					}
				}
				wordCount := fileProcessor.CountWords(text)
				return swarmgo.Result{
					Data: strconv.Itoa(wordCount),
				}
			},
		},
	}

	// Create a new agent with file analysis capabilities
	agent := &swarmgo.Agent{
		Name: "FileAnalyzer",
		Instructions: `You are a file analysis assistant. Follow these steps in order:

1. Use read_file to get the file content
2. Use count_words to count the words
3. Analyze the content and provide:
   - Word count and statistics
   - Key themes and topics
   - Structure and organization
   - Main points

Wait for each tool's result before proceeding.`,
		Functions: functions,
		Model:     "gpt-4o", // Using standard GPT-4
	}

	// Create a new swarm instance
	swarm := swarmgo.NewSwarm(apiKey, llm.OpenAI)

	// Create a custom stream handler
	handler := &CustomStreamHandler{}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// Create the analysis request
	messages := []llm.Message{
		{
			Role:    llm.RoleUser,
			Content: fmt.Sprintf("Please analyze the file at %s. First read it using read_file, then count words with count_words.", readmePath),
		},
	}

	fmt.Printf("Debug: Starting analysis with model %s\n", agent.Model)
	fmt.Printf("Debug: Reading file: %s\n", readmePath)

	// Start streaming analysis
	if err := swarm.StreamingResponse(ctx, agent, messages, nil, "", handler, false); err != nil {
		log.Fatalf("Error in streaming response: %v", err)
	}
}
