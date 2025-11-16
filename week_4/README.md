# Blog Generation Application

An intelligent blog generation application powered by LangGraph and LLMs (Large Language Models). This application leverages Groq and OpenAI to automatically generate high-quality blog content with optional language translation capabilities.

## Overview

This project my submission for the Week 4 bootcamp project that demonstrates the use of **LangGraph** for building stateful AI workflows. It provides a Streamlit-based UI for generating blog content in multiple languages using either Groq or OpenAI as the LLM backbone.

## Features

### 🎯 Use Cases

1. **Blog Generation with Topic**
   - Generate SEO-friendly blog titles based on a given topic
   - Create detailed, well-structured blog content with Markdown formatting

2. **Blog Generation with Language Translation**
   - Generate blog content in English
   - Validate requested language
   - Automatically translate the blog to the specified language

### 🤖 LLM Support

- **Groq**: Fast, open-source LLM inference platform
- **OpenAI**: GPT-based language models (GPT-3.5, GPT-4, etc.)

### 🎨 User Interface

- Built with **Streamlit** for an interactive, web-based experience
- Configurable sidebar for LLM selection, model selection, and API key input
- Support for multiple use cases and language selection
- Real-time chat interface for topic input

## Project Structure

```
week_4/
├── main.py                          # Entry point for the Streamlit application
├── pyproject.toml                   # Project dependencies and configuration
├── README.md                        # This file
└── src/
    ├── graphs/
    │   └── graph.py                 # Graph builder using LangGraph for workflow orchestration
    ├── models/
    │   ├── groq.py                  # Groq LLM model configuration
    │   └── openai.py                # OpenAI LLM model configuration
    ├── nodes/
    │   └── blog.py                  # Blog generation nodes (title, content, translation)
    ├── states/
    │   └── blog.py                  # State definitions for blog generation workflow
    └── ui/
        ├── load_ui.py               # Streamlit UI initialization and controls
        ├── display_result.py         # Result display formatting
        └── config/
            ├── ui_config_file.py    # UI configuration classes
            └── ui_config_file.ini    # Configuration settings
```

## Technical Architecture

### LangGraph Workflow

The application uses **LangGraph** to create two distinct workflows:

#### 1. Topic-Based Blog Generation
```
START → Title Creation → Content Generation → END
```

#### 2. Language Translation Workflow
```
START → Language Verification → Title Creation → Content Generation → Translation → END
```

### Key Components

- **GraphBuilder**: Orchestrates the workflow and manages node connections
- **BlogNode**: Contains the logic for title creation, content generation, language validation, and translation
- **BlogState**: Manages the state of the blog generation process using TypedDict
- **LLM Models**: Abstractions for Groq and OpenAI integration

## Installation

### Prerequisites

- Python 3.13 or higher
- API keys for either Groq or OpenAI (or both)

### Setup

1. Clone or download this project
2. Install dependencies:

```bash
uv sync
```

Or install dependencies manually:

```bash
uv add langchain-groq>=1.0.1 \
            langchain-openai>=1.0.3 \
            langgraph>=1.0.3 \
            pydantic>=2.12.4 \
            python-dotenv>=1.2.1 \
            streamlit-chat>=0.1.1 \
            watchdog>=6.0.0
```

## Usage

Run the Streamlit application:

```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501/`

### How to Use

1. **Select LLM Provider**: Choose between Groq or OpenAI from the sidebar
2. **Enter API Key**: Provide your API key for the selected provider
3. **Select Use Case**: Choose between:
   - Blog Generation with Topic
   - Blog Generation with Language Translation
4. **Provide Input**: 
   - For topic-based generation: Enter a blog topic
   - For language translation: Enter a topic and specify the target language
5. **View Results**: The generated blog with title and content will be displayed

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain-groq` | >=1.0.1 | Groq LLM integration |
| `langchain-openai` | >=1.0.3 | OpenAI LLM integration |
| `langgraph` | >=1.0.3 | Workflow orchestration and state management |
| `pydantic` | >=2.12.4 | Data validation and modeling |
| `python-dotenv` | >=1.2.1 | Environment variable management |
| `streamlit-chat` | >=0.1.1 | Chat interface components |
| `watchdog` | >=6.0.0 | File system event monitoring |

## Configuration

Configuration settings can be modified in `src/ui/config/ui_config_file.ini`:

- LLM options
- OpenAI and Groq model selections
- Use case definitions
- UI page title and styling

## API Keys

### Groq API Key
Get your free API key from [Groq Console](https://console.groq.com)

### OpenAI API Key
Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## Output Format

The application generates blog content in **Markdown format** for:
- Clean formatting
- Easy publication to various platforms
- Professional appearance
- SEO-friendly structure

## Error Handling

The application includes robust error handling for:
- Missing or invalid API keys
- LLM initialization failures
- Invalid language specification
- UI loading failures

All errors are displayed to the user in the Streamlit interface with clear, actionable messages.

## Notes

- This project is part of the GenAI Bootcamp - Week 4
- The application is designed for educational and demonstration purposes
- Ensure API keys are kept secure and never committed to version control
