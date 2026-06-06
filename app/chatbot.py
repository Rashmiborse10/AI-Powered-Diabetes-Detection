"""
AI Health Assistant Module
- NLP for natural language input processing
- RAG Chatbot using LangChain and FAISS for healthcare guidance
"""

import streamlit as st
import pandas as pd
import re
from typing import Dict, Optional, List, Tuple
import os
from pathlib import Path

from app.health_guidance import (
    guidance_to_markdown,
    get_risk_guidance,
    knowledge_corpus,
    normalize_text,
)

# LangChain imports
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Groq LLM import (free API, open-source models)
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ============================================================
# NLP PROCESSOR FOR NATURAL LANGUAGE INPUT
# ============================================================

class DiabetesNLPProcessor:
    """
    Extracts health parameters from natural language input
    for diabetes risk assessment.
    """
    
    def __init__(self):
        self.patterns = {
            'age': [
                r'(?:i am|i\'m|age is|aged?)\s*(\d+)',
                r'(\d+)\s*(?:years? old|yo|y\.?o\.?)',
                r'age[:\s]+(\d+)',
                r'(\d+)\s*(?:years?|yrs?)\s*(?:of age)?',
            ],
            'glucose': [
                r'(?:glucose|blood sugar|sugar level)[:\s]*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:mg/?dl|mg/dl)?\s*(?:glucose|blood sugar)',
                r'fasting[:\s]*(\d+(?:\.\d+)?)',
            ],
            'insulin': [
                r'(?:insulin)[:\s]*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:mu/l|μu/ml|iu/ml)?\s*insulin',
            ],
            'bmi': [
                r'(?:bmi|body mass index)[:\s]*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:bmi|body mass)',
            ],
            'pregnancies': [
                r'(?:pregnancies|pregnant|pregnancy)[:\s]*(\d+)',
                r'(\d+)\s*(?:pregnancies|times pregnant)',
                r'(?:been pregnant|gave birth)\s*(\d+)\s*times?',
                r'(?:never been pregnant|no pregnancies)',  # Handle 0
            ],
            'weight': [
                r'(?:weight|weigh)[:\s]*(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)?',
                r'(\d+(?:\.\d+)?)\s*(?:kg|kilograms?|lbs?|pounds?)',
            ],
            'height': [
                r'(?:height|tall)[:\s]*(\d+(?:\.\d+)?)\s*(?:cm|m|meters?|centimeters?)?',
                r'(\d+)\s*(?:\'|feet?|ft)\s*(\d+)?\s*(?:"|inches?|in)?',  # feet and inches
                r'(\d+(?:\.\d+)?)\s*(?:cm|centimeters?|m|meters?)',
            ],
        }
        
        # Feature constraints based on your model
        self.constraints = {
            'pregnancies': (0, 20),
            'glucose': (0, 250),
            'insulin': (0, 1000),
            'bmi': (0.0, 100.0),
            'age': (0, 100),
        }
    
    def extract_value(self, text: str, feature: str) -> Optional[float]:
        """Extract a single feature value from text."""
        text_lower = text.lower()
        
        # Special case for "never pregnant" or "no pregnancies"
        if feature == 'pregnancies':
            if re.search(r'(?:never|no|0)\s*(?:been\s+)?pregnan', text_lower):
                return 0.0
        
        for pattern in self.patterns.get(feature, []):
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    # Handle feet/inches for height
                    if feature == 'height' and len(match.groups()) == 2 and match.group(2):
                        feet = float(match.group(1))
                        inches = float(match.group(2)) if match.group(2) else 0
                        return (feet * 30.48) + (inches * 2.54)  # Convert to cm
                    return float(match.group(1))
                except (ValueError, TypeError):
                    continue
        return None
    
    def calculate_bmi(self, weight: float, height: float, weight_unit: str = 'kg', height_unit: str = 'cm') -> float:
        """Calculate BMI from weight and height."""
        # Convert to kg if needed
        if weight_unit == 'lbs':
            weight = weight * 0.453592
        
        # Convert height to meters
        if height_unit == 'cm':
            height = height / 100
        elif height_unit == 'ft':
            height = height * 0.3048
        
        if height > 0:
            return round(weight / (height ** 2), 1)
        return 0.0
    
    def parse_input(self, text: str) -> Dict[str, Optional[float]]:
        """Parse natural language input and extract all health parameters."""
        extracted = {}
        
        for feature in ['age', 'glucose', 'insulin', 'bmi', 'pregnancies']:
            value = self.extract_value(text, feature)
            if value is not None:
                # Apply constraints
                min_val, max_val = self.constraints[feature]
                value = max(min_val, min(max_val, value))
            extracted[feature] = value
        
        # If BMI not provided but weight and height are, calculate it
        if extracted.get('bmi') is None:
            weight = self.extract_value(text, 'weight')
            height = self.extract_value(text, 'height')
            
            if weight and height:
                # Detect units
                weight_unit = 'lbs' if 'lb' in text.lower() or 'pound' in text.lower() else 'kg'
                height_unit = 'ft' if ('ft' in text.lower() or 'feet' in text.lower() or "'" in text) else 'cm'
                
                extracted['bmi'] = self.calculate_bmi(weight, height, weight_unit, height_unit)
        
        return extracted
    
    def get_missing_features(self, extracted: Dict) -> List[str]:
        """Get list of features that weren't extracted."""
        required = ['age', 'glucose', 'insulin', 'bmi', 'pregnancies']
        return [f for f in required if extracted.get(f) is None]
    
    def create_dataframe(self, extracted: Dict, defaults: Dict = None) -> pd.DataFrame:
        """Create DataFrame for model prediction with extracted values and defaults."""
        defaults = defaults or {
            'Pregnancies': 1,
            'Glucose': 100,
            'Insulin': 100,
            'BMI': 25.0,
            'Age': 30
        }
        
        data = {
            'Pregnancies': extracted.get('pregnancies') or defaults['Pregnancies'],
            'Glucose': extracted.get('glucose') or defaults['Glucose'],
            'Insulin': extracted.get('insulin') or defaults['Insulin'],
            'BMI': extracted.get('bmi') or defaults['BMI'],
            'Age': extracted.get('age') or defaults['Age'],
        }
        
        return pd.DataFrame([data])


# ============================================================
# DIABETES KNOWLEDGE BASE FOR RAG
# ============================================================

DIABETES_KNOWLEDGE_BASE = """
# Diabetes Overview

## What is Diabetes?
Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels (hyperglycemia) resulting from defects in insulin secretion, insulin action, or both. The condition affects how your body converts food into energy.

## Types of Diabetes

### Type 1 Diabetes
- Autoimmune condition where the immune system attacks insulin-producing beta cells in the pancreas
- Usually diagnosed in children and young adults (but can occur at any age)
- Accounts for approximately 5-10% of all diabetes cases
- Requires lifelong insulin therapy
- Cannot be prevented with current medical knowledge
- Symptoms often appear suddenly and can include extreme thirst, frequent urination, unexplained weight loss, and fatigue

### Type 2 Diabetes
- The body becomes resistant to insulin or doesn't produce enough insulin
- Most common form, accounting for 90-95% of diabetes cases
- Often associated with lifestyle factors: obesity, physical inactivity, poor diet
- Genetics also play a significant role
- Can often be managed with lifestyle changes, oral medications, and sometimes insulin
- May develop gradually over years with few or no symptoms initially
- Risk increases with age, especially after 45

### Gestational Diabetes
- Develops during pregnancy in women who didn't have diabetes before
- Usually resolves after childbirth
- Increases risk of developing Type 2 diabetes later in life
- Requires careful monitoring during pregnancy to protect both mother and baby

## Key Risk Factors for Type 2 Diabetes

### Non-Modifiable Risk Factors
- Age (risk increases after 45)
- Family history of diabetes
- Ethnicity (higher risk in African American, Hispanic, Native American, Asian American populations)
- History of gestational diabetes
- Polycystic ovary syndrome (PCOS)

### Modifiable Risk Factors
- Overweight or obesity (BMI ≥ 25)
- Physical inactivity
- High blood pressure (≥140/90 mmHg)
- Abnormal cholesterol levels
- Poor diet high in processed foods and sugar
- Smoking

## Understanding Key Health Metrics

### Blood Glucose Levels
- Normal fasting glucose: Less than 100 mg/dL
- Prediabetes: 100-125 mg/dL (fasting)
- Diabetes: 126 mg/dL or higher (fasting)
- Random blood sugar of 200 mg/dL or higher suggests diabetes

### Body Mass Index (BMI)
- Underweight: Less than 18.5
- Normal weight: 18.5-24.9
- Overweight: 25-29.9
- Obese: 30 or higher
- Higher BMI increases diabetes risk significantly

### Insulin Levels
- Fasting insulin: Normal range is typically 2-25 mIU/L
- Higher fasting insulin may indicate insulin resistance
- Insulin resistance is a precursor to Type 2 diabetes

### HbA1c (Glycated Hemoglobin)
- Normal: Below 5.7%
- Prediabetes: 5.7% to 6.4%
- Diabetes: 6.5% or higher
- Reflects average blood sugar over past 2-3 months

## Common Symptoms of Diabetes

### Early Warning Signs
- Increased thirst (polydipsia)
- Frequent urination (polyuria)
- Extreme hunger (polyphagia)
- Unexplained weight loss
- Fatigue and weakness
- Blurred vision
- Slow-healing wounds or frequent infections
- Tingling or numbness in hands or feet
- Darkened skin patches (acanthosis nigricans)

## Complications of Uncontrolled Diabetes

### Cardiovascular Complications
- Heart disease (2-4 times higher risk)
- Stroke
- High blood pressure
- Atherosclerosis

### Eye Complications (Diabetic Retinopathy)
- Damage to blood vessels in the retina
- Can lead to blindness if untreated
- Regular eye exams are essential

### Kidney Complications (Diabetic Nephropathy)
- Kidney damage and potential kidney failure
- May require dialysis or kidney transplant
- Monitoring kidney function is crucial

### Nerve Damage (Diabetic Neuropathy)
- Peripheral neuropathy: numbness, tingling, pain in extremities
- Autonomic neuropathy: affects digestion, heart rate, bladder function
- Can lead to foot problems and amputations

### Foot Complications
- Poor circulation and nerve damage increase infection risk
- Foot ulcers and potential amputation
- Daily foot care is essential

## Prevention and Management

### Lifestyle Modifications
- Maintain healthy weight (BMI 18.5-24.9)
- Regular physical activity (150 minutes moderate exercise per week)
- Balanced diet rich in vegetables, whole grains, lean proteins
- Limit sugar and processed foods
- Quit smoking
- Moderate alcohol consumption
- Manage stress
- Get adequate sleep (7-9 hours)

### Dietary Recommendations
- Focus on low glycemic index foods
- Include fiber-rich foods (vegetables, legumes, whole grains)
- Choose healthy fats (olive oil, nuts, avocados)
- Limit saturated fats and trans fats
- Control portion sizes
- Eat at regular intervals
- Stay hydrated with water

### Exercise Benefits
- Improves insulin sensitivity
- Helps control blood sugar levels
- Aids in weight management
- Reduces cardiovascular risk
- Improves mental health
- Recommended: combination of aerobic and resistance training

### Monitoring and Testing
- Regular blood glucose monitoring
- HbA1c test every 3-6 months
- Annual eye exams
- Regular foot exams
- Blood pressure monitoring
- Cholesterol checks
- Kidney function tests

## When to Seek Medical Help

### Emergency Symptoms
- Very high blood sugar (above 300 mg/dL)
- Diabetic ketoacidosis symptoms: fruity breath, nausea, vomiting, confusion
- Signs of hypoglycemia: shakiness, sweating, confusion, rapid heartbeat
- Chest pain or difficulty breathing
- Severe dehydration

### Schedule a Doctor Visit If
- You have risk factors for diabetes
- You experience any diabetes symptoms
- Your blood sugar readings are consistently abnormal
- You need help managing your diabetes
- You're planning to become pregnant

## Pregnancy and Diabetes

### Gestational Diabetes Risk Factors
- Previous gestational diabetes
- Family history of diabetes
- Overweight or obese
- Age over 25
- Previous delivery of baby over 9 pounds
- Polycystic ovary syndrome

### Management During Pregnancy
- Regular blood sugar monitoring
- Healthy eating plan
- Safe physical activity
- Medication if needed
- More frequent prenatal visits

## Mental Health and Diabetes

### Emotional Impact
- Diabetes distress is common
- Higher risk of depression and anxiety
- Importance of mental health support
- Stress can affect blood sugar levels

### Coping Strategies
- Join support groups
- Practice stress management
- Maintain social connections
- Seek professional help when needed
"""


# ============================================================
# RAG CHATBOT CLASS
# ============================================================

# Prompt template for the RAG chain
RAG_PROMPT_TEMPLATE = """You are a knowledgeable and friendly health assistant with deep expertise in diabetes.
You can answer ANY question — whether it's about diabetes, general health, or completely unrelated topics like history, science, cooking, or technology.

When the question is about diabetes or health, use the retrieved context below to give accurate, grounded answers.
When the question is unrelated to diabetes, answer it helpfully from your general knowledge.

Always be clear, concise, and compassionate. For any medical advice, remind the user to consult a healthcare professional.

{pred_info}

Retrieved context (use if relevant):
{context}

Question: {question}

Answer:"""


class DiabetesRAGChatbot:
    """
    RAG-based chatbot using LangChain + Ollama (free, local, open-source).
    Handles both diabetes-specific and general questions.
    """

    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None
        self.embeddings = None
        self.is_initialized = False
        self._use_llm = False  # True when Ollama is available

    def initialize(self):
        """Initialize FAISS retriever and Groq LLM."""
        if not LANGCHAIN_AVAILABLE:
            st.warning("LangChain packages not found. Install with: pip install langchain-huggingface langchain-community faiss-cpu")
            self.is_initialized = True
            return True
        try:
            with st.spinner("Initializing AI Health Assistant..."):
                    # ── 1. Embeddings + FAISS ──────────────────────────────
                self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                    )
                index_dir = Path("data/diabetes_faiss")
                if index_dir.exists():
                    self.vectorstore = FAISS.load_local(
                    str(index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " "],
                 )
                    texts = text_splitter.split_text("\n\n".join(knowledge_corpus()))
                    self.vectorstore = FAISS.from_texts(texts=texts, embedding=self.embeddings)
                    index_dir.mkdir(parents=True, exist_ok=True)
                    self.vectorstore.save_local(str(index_dir))
                self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            )
            # ── 2. Groq LLM ───────────────────────────────────────
                if GROQ_AVAILABLE:
                    try:
                        self.llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        temperature=0.3,
                        api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", ""),
                    )
                        self._use_llm = True
                    except Exception as e:
                        st.warning(
                        f"Groq could not be initialized ({e}). "
                        "Check your GROQ_API_KEY environment variable. "
                        "Get a free key at console.groq.com. "
                        "Falling back to rule-based responses."
                    )
                else:
                    st.info(
                    "langchain-groq not installed. "
                    "Run: pip install langchain-groq — then add GROQ_API_KEY to your environment. "
                    "Using rule-based responses for now."
                )
            # ── 3. LangChain RAG chain ─────────────────────────────
                if self._use_llm:
                    prompt = PromptTemplate(
                    input_variables=["context", "question", "pred_info"],
                    template=RAG_PROMPT_TEMPLATE,
                )
                    self.rag_chain = prompt | self.llm
                self.is_initialized = True
                return True
        except Exception as e:
            st.warning(f"RAG system could not be initialized ({e}). Using rule-based responses.")
            self.is_initialized = True
            return True

    # ── Retrieval ──────────────────────────────────────────────────────────

    def get_relevant_context(self, query: str) -> str:
        """Retrieve relevant chunks from FAISS."""
        if not self.is_initialized or self.retriever is None:
            return ""
        try:
            docs = self.retriever.invoke(normalize_text(query))
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return ""

    # ── Main entry point ───────────────────────────────────────────────────

    def generate_response(self, query: str, prediction_context: Dict = None) -> str:
        """Generate a response for any question, with optional prediction context."""
        if not self.is_initialized:
            return "The AI assistant is not initialized. Please refresh the page."

        # Build prediction info string
        pred_info = ""
        if prediction_context:
            prob = prediction_context.get("probability", 0)
            is_diabetic = prediction_context.get("is_diabetes", False)
            inputs = prediction_context.get("inputs", {})
            pred_info = (
                f"Current User Assessment:\n"
                f"- Diabetes Risk Probability: {prob}%\n"
                f"- Risk Level: {'HIGH RISK' if is_diabetic else 'Lower Risk'}\n"
                f"- Inputs: Age={inputs.get('Age','N/A')}, "
                f"Glucose={inputs.get('Glucose','N/A')} mg/dL, "
                f"Insulin={inputs.get('Insulin','N/A')}, "
                f"BMI={inputs.get('BMI','N/A')}, "
                f"Pregnancies={inputs.get('Pregnancies','N/A')}"
            )

        # Prediction-result questions → structured response (always)
        query_lower = query.lower()
        if pred_info and any(w in query_lower for w in ["my", "result", "prediction", "risk", "score", "assessment"]):
            return self._format_prediction_response(pred_info, prediction_context)

        # Try Ollama RAG chain first
        if self._use_llm and self.rag_chain:
            try:
                context = self.get_relevant_context(query)
                response = self.rag_chain.invoke({
                    "context": context or "No specific context retrieved.",
                    "question": query,
                    "pred_info": pred_info or "",
                })
                if hasattr(response, "content"):
                    return response.content.strip()
                return str(response).strip()
            except Exception as e:
                # Fall through to rule-based on error
                pass

        # Fallback: rule-based responses (works without Ollama)
        context = self.get_relevant_context(query)
        return self._rule_based_response(query_lower, context, pred_info)

    # ── Prediction result formatter ────────────────────────────────────────

    def _format_prediction_response(self, pred_info: str, prediction_context: Dict) -> str:
        """Structured response for prediction-related questions."""
        try:
            prob = float(prediction_context.get("probability", 0))
        except (TypeError, ValueError):
            prob = 0.0

        is_high_risk = prediction_context.get("is_diabetes", False)
        inputs = prediction_context.get("inputs", {})
        guidance = get_risk_guidance(prob, is_high_risk, inputs)
        return (
            f"**Your Diabetes Risk Assessment**\n\n"
            f"{pred_info}\n\n"
            f"{guidance_to_markdown(guidance)}\n\n"
            "**Important:** This is a screening tool for educational purposes only. "
            "A clinician can confirm diagnosis and tailor treatment."
        )

    # ── Rule-based fallback (no Ollama needed) ─────────────────────────────

    def _rule_based_response(self, query_lower: str, context: str, pred_info: str) -> str:
        """Keyword-based fallback when Ollama is unavailable."""
        topic_map = [
            (["symptom", "sign", "feel", "experience"],                         "symptoms"),
            (["prevent", "avoid", "reduce risk", "lower risk"],                 "prevention"),
            (["eat", "food", "diet", "nutrition", "meal"],                      "diet"),
            (["exercise", "physical", "activity", "workout"],                   "exercise"),
            (["complication", "problem", "damage", "effect"],                   "complications"),
            (["type 1", "type1", "type one"],                                   "type1"),
            (["type 2", "type2", "type two"],                                   "type2"),
            (["glucose", "blood sugar", "sugar level"],                         "glucose"),
            (["bmi", "weight", "body mass"],                                    "bmi"),
            (["insulin"],                                                        "insulin"),
            (["pregnant", "pregnancy", "gestational"],                          "pregnancy"),
        ]

        for keywords, topic in topic_map:
            if any(kw in query_lower for kw in keywords):
                return self._topic_response(topic)

        # General diabetes context from FAISS
        if context:
            return (
                f"Based on the diabetes knowledge base:\n\n{context[:1500]}\n\n"
                "**Important:** This information is for educational purposes only. "
                "Please consult a healthcare professional for personalized medical advice."
            )

        return (
            "I can help with diabetes symptoms, prevention, diet, exercise, blood glucose, "
            "BMI, insulin, complications, and your prediction results.\n\n"
            "**Tip:** Add a GROQ_API_KEY environment variable (free at console.groq.com) "
            "to enable full AI responses for any question."
        )

    def _topic_response(self, topic: str) -> str:
        """Return a canned response for a known diabetes topic."""
        responses = {
            "symptoms": (
                "**Common Diabetes Symptoms**\n\n"
                "- Increased thirst (polydipsia)\n"
                "- Frequent urination (polyuria)\n"
                "- Extreme hunger (polyphagia)\n"
                "- Unexplained weight loss\n"
                "- Fatigue and weakness\n"
                "- Blurred vision\n"
                "- Slow-healing wounds\n"
                "- Tingling in hands or feet\n"
                "- Darkened skin patches\n\n"
                "If you experience several of these, consult a healthcare provider."
            ),
            "prevention": (
                "**Diabetes Prevention**\n\n"
                "- Maintain healthy weight (BMI 18.5–24.9)\n"
                "- Exercise 150+ min/week\n"
                "- Eat balanced, low-sugar diet\n"
                "- Quit smoking, limit alcohol\n"
                "- Manage stress, sleep 7–9 hours\n"
                "- Annual blood sugar check if over 45"
            ),
            "diet": (
                "**Diet Recommendations**\n\n"
                "Include: vegetables, whole grains, lean proteins, healthy fats, fiber.\n\n"
                "Limit: sugary drinks, refined carbs, processed foods, saturated fats."
            ),
            "exercise": (
                "**Exercise Recommendations**\n\n"
                "- 150 min/week aerobic (walking, swimming, cycling)\n"
                "- Resistance training 2–3×/week\n"
                "- Break up sitting every 30 min\n"
                "- Consult your doctor before starting a new program"
            ),
            "complications": (
                "**Diabetes Complications**\n\n"
                "Cardiovascular: heart disease, stroke, high blood pressure\n"
                "Eyes: diabetic retinopathy → blindness\n"
                "Kidneys: diabetic nephropathy → kidney failure\n"
                "Nerves: neuropathy → numbness, pain\n"
                "Feet: ulcers, infections, amputation risk\n\n"
                "Good glucose control greatly reduces these risks."
            ),
            "type1": (
                "**Type 1 Diabetes**\n\n"
                "Autoimmune — immune system destroys pancreatic beta cells.\n"
                "Usually diagnosed young. Requires lifelong insulin therapy. Cannot be prevented."
            ),
            "type2": (
                "**Type 2 Diabetes**\n\n"
                "Body becomes insulin-resistant or produces insufficient insulin.\n"
                "90–95% of cases. Linked to lifestyle + genetics. Often preventable/manageable."
            ),
            "glucose": (
                "**Blood Glucose Levels**\n\n"
                "Normal fasting: < 100 mg/dL\n"
                "Prediabetes: 100–125 mg/dL\n"
                "Diabetes: ≥ 126 mg/dL\n"
                "Random ≥ 200 mg/dL also suggests diabetes."
            ),
            "bmi": (
                "**BMI Categories**\n\n"
                "Underweight: < 18.5 | Normal: 18.5–24.9 | Overweight: 25–29.9 | Obese: ≥ 30\n\n"
                "Higher BMI strongly increases Type 2 diabetes risk. Even 5–7% weight loss helps."
            ),
            "insulin": (
                "**Insulin**\n\n"
                "Pancreatic hormone that moves glucose into cells.\n"
                "Normal fasting: 2–25 mIU/L. High fasting insulin → insulin resistance → T2D risk."
            ),
            "pregnancy": (
                "**Gestational Diabetes**\n\n"
                "Develops during pregnancy. Usually resolves after birth but raises future T2D risk.\n"
                "Managed with diet, exercise, and sometimes medication."
            ),
        }
        text = responses.get(topic, "I don't have specific information on that topic.")
        return text + "\n\n**Disclaimer:** Educational purposes only. Consult a healthcare professional for personal advice."


# ============================================================
# STREAMLIT APP INTERFACE
# ============================================================

def app():
    """Main chatbot application interface."""
    
    st.markdown("---")
    st.markdown("## AI Health Assistant")
    st.markdown("*Powered by NLP and RAG (LangChain + FAISS)*")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'nlp_processor' not in st.session_state:
        st.session_state.nlp_processor = DiabetesNLPProcessor()
    if 'rag_chatbot' not in st.session_state:
        st.session_state.rag_chatbot = DiabetesRAGChatbot()
        st.session_state.rag_chatbot.initialize()
    if 'nlp_extracted' not in st.session_state:
        st.session_state.nlp_extracted = {}
    
    # Create tabs
    tab1, tab2 = st.tabs(["Chat Assistant", "Natural Language Input"])
    
    # Tab 1: Chat Assistant (RAG)
    with tab1:
        st.markdown("### Ask me about diabetes")
        st.markdown("I can answer questions about symptoms, risk factors, prevention, diet, exercise, and explain your prediction results.")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}")
                st.markdown("---")
        
        # Input area
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Your question:",
                key="chat_input",
                placeholder="e.g., What are the symptoms of diabetes?"
            )
        with col2:
            send_button = st.button("Send", key="send_btn")
        
        if send_button and user_input:
            # Get prediction context if available
            pred_context = st.session_state.get('prediction_context', None)
            
            # Generate response
            response = st.session_state.rag_chatbot.generate_response(
                user_input, 
                pred_context
            )
            
            # Update chat history
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            
            st.rerun()
        
        # Quick question buttons
        st.markdown("**Quick Questions:**")
        quick_cols = st.columns(3)
        quick_questions = [
            "What are diabetes symptoms?",
            "How can I prevent diabetes?",
            "What should I eat?",
            "Explain my results",
            "What is normal blood sugar?",
            "How does BMI affect risk?"
        ]
        
        for i, question in enumerate(quick_questions):
            with quick_cols[i % 3]:
                if st.button(question, key=f"quick_{i}"):
                    pred_context = st.session_state.get('prediction_context', None)
                    response = st.session_state.rag_chatbot.generate_response(question, pred_context)
                    st.session_state.chat_history.append({'role': 'user', 'content': question})
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                    st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Tab 2: Natural Language Input (NLP)
    with tab2:
        st.markdown("### Describe Your Health in Natural Language")
        st.markdown("Instead of using the sidebar, you can describe your health parameters in plain English.")
        
        st.info("""
        **Example inputs:**
        - "I'm a 45 year old woman, my glucose level is 140 mg/dL, insulin is 85, BMI is 28.5, and I've been pregnant twice"
        - "Age 35, blood sugar 110, never been pregnant, BMI 24"
        - "I'm 52, my fasting glucose was 135, I weigh 180 lbs and I'm 5'8"
        """)
        
        nlp_input = st.text_area(
            "Describe your health parameters:",
            height=100,
            placeholder="Example: I'm 45 years old, my glucose is 130, insulin 100, BMI 32, and I've had 2 pregnancies"
        )
        
        if st.button("Extract Parameters", key="extract_btn"):
            if nlp_input:
                extracted = st.session_state.nlp_processor.parse_input(nlp_input)
                st.session_state.nlp_extracted = extracted
                
                st.markdown("### Extracted Parameters:")
                
                cols = st.columns(5)
                features = ['age', 'glucose', 'insulin', 'bmi', 'pregnancies']
                labels = ['Age', 'Glucose (mg/dL)', 'Insulin', 'BMI', 'Pregnancies']
                
                for i, (feature, label) in enumerate(zip(features, labels)):
                    with cols[i]:
                        value = extracted.get(feature)
                        if value is not None:
                            st.success(f"**{label}**\n\n{value}")
                        else:
                            st.warning(f"**{label}**\n\nNot found")
                
                missing = st.session_state.nlp_processor.get_missing_features(extracted)
                if missing:
                    st.warning(f"Missing parameters: {', '.join(missing)}. Default values will be used for prediction.")
        
        # Show extracted values and allow manual adjustment
        if st.session_state.nlp_extracted:
            st.markdown("### Adjust Values (if needed):")
            
            adj_cols = st.columns(5)
            adjusted = {}
            
            defaults = {'age': 30, 'glucose': 100, 'insulin': 100, 'bmi': 25.0, 'pregnancies': 1}
            
            with adj_cols[0]:
                adjusted['age'] = st.number_input(
                    "Age",
                    min_value=0, max_value=100,
                    value=int(st.session_state.nlp_extracted.get('age') or defaults['age']),
                    key="adj_age"
                )
            with adj_cols[1]:
                adjusted['glucose'] = st.number_input(
                    "Glucose",
                    min_value=0, max_value=250,
                    value=int(st.session_state.nlp_extracted.get('glucose') or defaults['glucose']),
                    key="adj_glucose"
                )
            with adj_cols[2]:
                adjusted['insulin'] = st.number_input(
                    "Insulin",
                    min_value=0, max_value=1000,
                    value=int(st.session_state.nlp_extracted.get('insulin') or defaults['insulin']),
                    key="adj_insulin"
                )
            with adj_cols[3]:
                adjusted['bmi'] = st.number_input(
                    "BMI",
                    min_value=0.0, max_value=100.0,
                    value=float(st.session_state.nlp_extracted.get('bmi') or defaults['bmi']),
                    key="adj_bmi"
                )
            with adj_cols[4]:
                adjusted['pregnancies'] = st.number_input(
                    "Pregnancies",
                    min_value=0, max_value=20,
                    value=int(st.session_state.nlp_extracted.get('pregnancies') or defaults['pregnancies']),
                    key="adj_pregnancies"
                )
            
            if st.button("Use These Values for Prediction", key="use_nlp_btn"):
                # Create DataFrame and store in session state for prediction
                input_df = pd.DataFrame([{
                    'Pregnancies': adjusted['pregnancies'],
                    'Glucose': adjusted['glucose'],
                    'Insulin': adjusted['insulin'],
                    'BMI': adjusted['bmi'],
                    'Age': adjusted['age']
                }])
                st.session_state.nlp_input_data = input_df
                st.success("Parameters saved. The prediction will use these values. Please refresh the page or re-run the prediction.")
                st.info("Note: To see the prediction with these values, the sidebar inputs will be overridden.")