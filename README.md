# Bible RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot for answering Bible-related questions, specifically using the King James Version (KJV). The chatbot uses open source LLMs from Ollama, ChromaDB as a vector store, and Streamlit for the user interface. It is meant to demonstrate the potential use of Large Language Models (LLMs) in ministry.

The project also serves as an experiment to establish the efficacy of open source, medium sized (~7B-10B params) LLMs in a potetal use case.

## Disclaimer:
- This chatbot is **very experimental**. Outputs and citations may not be 100% accurate, therefore it is not meant for widespread use, but rather as a basis for a broader discussion and a much more refined product that will bring value to Christians *and all* who are interested in exploring God's word.

- This chatbot **can not and will never** be a substitute for the ***Holy Spirit***, who makes all that is hidden and mysterious to be known.

- As the Bible is no ordinary document, but God Himself ***(John 1:1)***, there is a huge responsibility on the developer to ensure that the Bible is presented holistically, without any additions or omissions of any kind. ***(Revelation 22:18-19), (Matthew 18:6)***

### Features

- Question answering based on the King James Version of the Bible
- Corrective RAG for built-in fact checking, boosting answer contextuality and accuracy
- Streamlit-based user interface for easy interaction
- Dockerized setup for easy deployment and scalability

### Prerequisites

- Docker and Docker Compose
- Git (for cloning the repository)
- CPU with >=10GB memory (CPU only version)

### Project Structure

```
bible-rag-chatbot/
│
├── chatbot.py          # Main Streamlit application
├── preprocess.py       # Script to preprocess Bible text and create vector store
├── config.py           # Configuration settings
├── .env                # Environment variables
├── requirements.txt    # Python dependencies
├── Dockerfile          # Dockerfile for the Python application
└── docker-compose.yml  # Docker Compose configuration
```

### Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/scha-ron/secret-project.git
   cd bible-rag-chatbot
   ```

2. Create a `.env` file in the project root and set the `DB_PATH`:
   ```
   DB_PATH=/app/bible_vectorstore
   ```

3. Build and run the Docker containers:
   ```
   docker-compose up --build
   ```

4. Once the containers are running, you need to pull the required Ollama models in a separate terminal:
   ```
   docker-compose exec ollama ollama pull llama3
   docker-compose exec ollama ollama pull gemma2
   ```

5. Access the chatbot interface at `http://localhost:8501` in your web browser.

### Usage

- Open the Streamlit interface in your web browser.
- Type your Bible-related questions in the chat input.
- The chatbot will provide answers based on the KJV Bible, using RAG for accurate and contextual responses.

### Development

To modify the chatbot or add new features:

1. Make changes to the Python files as needed.
2. Rebuild and restart the Docker containers:
   ```
   docker-compose down
   docker-compose up --build
   ```
### Known issues:

- Long load times due to heavy resource usage by LLMs (cpu-only version)
- Long load and response times in follow up questions, such as 'tell me more'. Models tend to hallucinate despite clear instructions in prompt template. This also includes the references/citations shown at the end of the responses.

### Troubleshooting

- If you encounter issues with Ollama models, ensure they are properly pulled and available in the Ollama container.
- Check Docker logs for any error messages:
  ```
  docker-compose logs
  ```

## Contributing

Contributions to improve the chatbot are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### License

This project is licensed under the Apache License - see the LICENSE file for details.

### Acknowledgments

- Bible text source: [Bible SuperSearch API](https://api.biblesupersearch.com/)



