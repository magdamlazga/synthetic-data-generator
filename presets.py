"""Ready-made datasets to start from. Columns are [name, type, description, options or range]."""

PRESETS = {
    "Support chatbot test messages": {
        "description": "Messages that customers send to the support chatbot of an online electronics and home goods "
                       "shop, used to test whether the chatbot understands the request and responds correctly.",
        "edge": 20,
        "columns": [
            ["customer_message", "text", "The message exactly as the customer typed it, 1 to 5 sentences", ""],
            ["intent", "category", "What the customer wants", "order_status, refund, cancel_order, delivery_problem, product_question, complaint, account_access"],
            ["mood", "category", "How the customer feels", "calm, confused, frustrated, angry"],
            ["includes_order_number", "boolean", "Whether the message contains an order number like BC-1234", ""],
            ["expected_bot_action", "text", "What a good support bot should do next, in one sentence", ""],
        ],
    },
    "Bug reports": {
        "description": "Bug reports filed by testers and users for a web-based project management app "
                       "(boards, tasks, comments, file uploads, notifications, billing).",
        "edge": 10,
        "columns": [
            ["title", "text", "Short bug title", ""],
            ["steps_to_reproduce", "text", "Numbered steps to reproduce", ""],
            ["expected_result", "text", "What should happen", ""],
            ["actual_result", "text", "What happens instead", ""],
            ["severity", "category", "Impact of the bug", "critical, major, minor, trivial"],
            ["component", "category", "Part of the app", "boards, tasks, comments, uploads, notifications, billing, login"],
            ["browser", "category", "Where it was found", "Chrome, Firefox, Safari, Edge, iOS app, Android app"],
        ],
    },
    "E-commerce orders": {
        "description": "Orders from an online shop that sells electronics, clothing and home goods across Europe, "
                       "used as test data for an order management system.",
        "edge": 15,
        "columns": [
            ["order_id", "text", "Order number like BC-48213", ""],
            ["customer_name", "text", "Full name of the customer", ""],
            ["country", "category", "Shipping country", "Poland, Germany, France, Spain, Italy, Netherlands, Sweden"],
            ["product", "text", "Product name as shown in the shop", ""],
            ["quantity", "integer", "Number of items", "1..10"],
            ["unit_price_eur", "number", "Price of one item in euros", "2..2500"],
            ["order_date", "date", "Date the order was placed, in 2025 or 2026", ""],
            ["status", "category", "Current order status", "pending, paid, shipped, delivered, returned, cancelled"],
            ["is_gift", "boolean", "Whether gift wrapping was requested", ""],
        ],
    },
    "Sign-up form inputs": {
        "description": "What people type into the sign-up form of a banking app, used to test form validation. "
                       "Include both inputs that should be accepted and inputs that should be rejected.",
        "edge": 50,
        "columns": [
            ["full_name", "text", "Name as typed by the user", ""],
            ["email", "text", "Email address as typed by the user", ""],
            ["password", "text", "Password as typed by the user", ""],
            ["birth_date", "date", "Date of birth", ""],
            ["should_be_accepted", "boolean", "Whether a correct form should accept this sign-up", ""],
            ["reason", "text", "Why it should be accepted or rejected, in one sentence", ""],
        ],
    },
    "Product reviews": {
        "description": "Customer reviews of kitchen appliances (coffee machines, blenders, air fryers, kettles, toasters) "
                       "on a shopping website, used to test a review sentiment classifier.",
        "edge": 10,
        "columns": [
            ["product", "text", "Product name with brand", ""],
            ["rating", "integer", "Star rating", "1..5"],
            ["review_text", "text", "The review as the customer wrote it", ""],
            ["sentiment", "category", "Overall sentiment of the review", "positive, neutral, negative, mixed"],
            ["verified_purchase", "boolean", "Whether the reviewer bought the product", ""],
        ],
    },
}

DEFAULT_PRESET = "Support chatbot test messages"
