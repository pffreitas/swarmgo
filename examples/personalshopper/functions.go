package main

import (
	"fmt"
	"math/rand"
	"strconv"
	"time"

	"github.com/pffreitas/swarmgo"
)

func refundItem(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	userIDStr, ok1 := args["user_id"].(string)
	itemIDStr, ok2 := args["item_id"].(string)

	if !ok1 || !ok2 {
		return swarmgo.Result{Data: "Please provide both user_id and item_id."}
	}

	userID, err1 := strconv.Atoi(userIDStr)
	itemID, err2 := strconv.Atoi(itemIDStr)
	if err1 != nil || err2 != nil {
		return swarmgo.Result{Data: "Invalid user_id or item_id format."}
	}

	conn := getConnection()

	var amount float64
	err := conn.QueryRow(`
		SELECT amount FROM PurchaseHistory
		WHERE user_id = ? AND item_id = ?;
	`, userID, itemID).Scan(&amount)

	if err != nil {
		return swarmgo.Result{Data: fmt.Sprintf("No purchase found for user ID %d and item ID %d.", userID, itemID)}
	}

	fmt.Printf("Refunding $%.2f to user ID %d for item ID %d.\n", amount, userID, itemID)
	fmt.Println("Refund initiated")

	return swarmgo.Result{Success: true, Data: "Refund initiated successfully."}
}

func notifyCustomer(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	userIDStr, ok1 := args["user_id"].(string)
	method, ok2 := args["method"].(string)

	if !ok1 || !ok2 {
		return swarmgo.Result{Data: "Please provide both user_id and method."}
	}

	userID, err1 := strconv.Atoi(userIDStr)
	if err1 != nil {
		return swarmgo.Result{Data: "Invalid user_id format."}
	}

	conn := getConnection()

	var email, phone string
	err := conn.QueryRow(`
		SELECT email, phone FROM Users WHERE user_id = ?;
	`, userID).Scan(&email, &phone)

	if err != nil {
		return swarmgo.Result{Data: fmt.Sprintf("User ID %d not found.", userID)}
	}

	if method == "email" && email != "" {
		fmt.Printf("Emailed customer %s a notification.\n", email)
		return swarmgo.Result{Data: fmt.Sprintf("Emailed customer %s a notification.", email)}
	} else if method == "phone" && phone != "" {
		fmt.Printf("Texted customer %s a notification.\n", phone)
		return swarmgo.Result{Data: fmt.Sprintf("Texted customer %s a notification.", phone)}
	} else {
		return swarmgo.Result{Data: fmt.Sprintf("No %s contact available for user ID %d.", method, userID)}
	}
}

func orderItem(args map[string]interface{}, contextVariables map[string]interface{}) swarmgo.Result {
	userIDStr, ok1 := args["user_id"].(string)
	productIDStr, ok2 := args["product_id"].(string)

	if !ok1 || !ok2 {
		return swarmgo.Result{Data: "Please provide both user_id and product_id."}
	}

	userID, err1 := strconv.Atoi(userIDStr)
	productID, err2 := strconv.Atoi(productIDStr)
	if err1 != nil || err2 != nil {
		return swarmgo.Result{Data: "Invalid user_id or product_id format."}
	}

	conn := getConnection()

	var productName string
	var price float64
	err := conn.QueryRow(`
		SELECT product_name, price FROM Products WHERE product_id = ?;
	`, productID).Scan(&productName, &price)

	if err != nil {
		return swarmgo.Result{Data: fmt.Sprintf("Product %d not found.", productID)}
	}

	dateOfPurchase := time.Now().Format("2006-01-02")
	itemID := rand.Intn(300) + 1

	fmt.Printf("Ordering product %s for user ID %d. The price is %.2f.\n", productName, userID, price)
	addPurchase(userID, dateOfPurchase, itemID, price)

	return swarmgo.Result{Data: fmt.Sprintf("Ordered %s for user ID %d.", productName, userID)}
}
