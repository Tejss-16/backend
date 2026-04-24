# 📊 JAVAX

A backend system that generates charts from dataset based on user prompts.

---

## 🚀 Setup Instructions

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🧠 Key Features

* 🔍 Natural language → data insights
* 📈 Automatic chart generation (bar, line, pie, etc.)
* 📊 Pivot tables & grouped summaries
* 🧾 KPI scorecards (top-level metrics)
* ⚡ Async + parallel processing for performance
* 🧠 Smart fallbacks when LLM fails
* 💾 Caching to avoid repeated LLM calls

---

## 🏗️ Architecture

The system follows a **layered MVC-inspired structure**:

```
app/
│
├── api/            # Routes (FastAPI endpoints)
|   ├── Routes.py
│  
├── services/       # Business logic (ChartGenerator)
|   ├── chart_service.py
│   
├── pipeline/       # Core processing pipeline
│   ├── llm_client.py
│   ├── normalizer.py
│   ├── transformer.py
│   ├── chart_builder.py
│   ├── table_builder.py
│   └── scorecard.py
│
├── schemas/        # Pydantic models
|   ├── chart_schema.py
│   
|
├── utils/          # Caching & helpers
|   ├── cache.py
│   ├── task_manager.py
│   
└── main.py         # Entry point
```

---

## 🔄 Flow

```
User Query + File
        ↓
API Route
        ↓
ChartGenerator (Service Layer)
        ↓
LLM → Chart/Table Config
        ↓
Normalization + Transformation
        ↓
Chart/Table/Scorecard Builders
        ↓
Final JSON Response
```

---

## ⚙️ Tech Stack

* **Backend:** FastAPI
* **Data Processing:** Pandas
* **LLM:** OpenRouter (Nemotron model)
* **Validation:** Pydantic
* **Async:** asyncio
* **Parallelism:** ThreadPoolExecutor

---

## ⚡ Performance Optimizations

### 1. Async LLM Calls

LLM requests are handled asynchronously to avoid blocking the API.

### 2. Thread-Based Parallelism

Heavy operations (charts, tables, scorecards) are executed using a shared thread pool:

```python
run_in_executor(EXECUTOR, ...)
```

### 3. Shared Executor

A global ThreadPoolExecutor avoids thread creation overhead.

### 4. Parallel Chart Building

Multiple charts are generated concurrently using threads.

### 5. Caching

Repeated queries on the same dataset are served instantly.

---

## 🚨 Key Challenges Solved

### ❌ Initial Issue

* Requests were processed sequentially
* Slow response due to blocking operations

### ✅ Fixes Implemented

* Converted pipeline to async
* Offloaded CPU-heavy tasks to threads
* Removed per-request thread pools
* Introduced shared executor

---

## ⚠️ Known Limitations

* LLM latency is the main bottleneck (external API)
* Pandas operations are CPU-bound (limited by Python GIL)
* Free-tier models may throttle concurrent requests

---

## 📦 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start server

```bash
uvicorn app.main:app --reload
```

### 3. Open API docs

```
http://127.0.0.1:8000/docs
```

---

## 💡 Summary

This project demonstrates:

* Real-world use of LLMs in data systems
* Async + parallel backend design
* Handling CPU + IO bottlenecks
* Building scalable data pipelines

---
