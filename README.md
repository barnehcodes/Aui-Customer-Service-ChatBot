# Aui-Customer-Service-ChatBot
# Milestone 1

University students are often discouraged by inefficiency in important academic processes like registration, admissions, and add/drop periods due to poor human support availability, especially during peak times. This generates inefficiencies, delays, and frustration. A 24/7 AI-powered Customer Service ChatBot can alleviate this through instant, precise, and consistent responses to FAQs, guiding students through procedures, and reducing the workload on human personnel.

Business Value of Using ML (Impact):

- Enhanced Student Experience: Real-time, round-the-clock support facilitates smoother educational processes, minimizing stress and improving satisfaction.
- Operational Efficiency: Reduces repetitive questions, allowing more resource-intensive tasks to be taken up by human resources and shortening response time.
- Scalability: Accommodates large numbers of queries during busy times without the need for additional staffing expenses.
- Cost Savings: Avoids the need for extensive human customer service personnel, lowering operational costs in the long term.
- Consistency: Provides accurate and uniform information, avoiding errors and miscommunication.


## AUI Chat (Human-In-The-loop)

This project combines the strengths of **automation** (via the chatbot) with **human oversight (”human agents”)**

### **Example Workflow**:

1. A student asks the chatbot, *“How do I appeal a grade?”*
2. The chatbot recognizes the query as complex and responds: *“This is a complex issue. Let me connect you with a human advisor.”*
3. The query is escalated to a human agent, who provides a detailed response.
4. The interaction is logged, and the chatbot’s performance is evaluated. If the chatbot could have handled the query better, the data is used to retrain the model.

![4. Example Workflow_ - visual selection (2).png](attachment:52ebcaf3-6c4e-4682-88c1-a17937d09f14:4._Example_Workflow__-_visual_selection_(2).png)

### Feasibility Study

now lets discuss the feasibility of this project there are a lot of aspects to consider:

**Data Availability and Curation**

*“A common theme with practically all of the university chatbots being reviewed is the small amounts of data that are being utilized, with the difficulty of access to training data being specifically mentioned by many authors.”*

- Also data amount and quality can vary depending on specific “university-zones” ( departments: SSE related data more then other schools …)

⇒ this might introduce **Bias.**

**Cost and Resource Constraints**

The financial cost and the currently on-hand resources poses a challenge in the flexibility of choices we can have in tools, base-models, deployment …, that said we ended up limiting to 

**Adoption and Effectiveness**

- While chatbot usage in universities is growing, many institutions have yet to fully realize their potential beyond simple FAQs.
- The feasibility of large-scale chatbot deployment depends on institutional willingness to invest in AI research, infrastructure, and data structuring.

### Related works

One of the main sources that are placed very close to the scope of this project is the research paper ‘**Building and Evaluating a Chatbot Using a University FAQs Dataset’ ([Said A. Salloum](), [Khaled Shaalan], [Azza Basiouni], [Ayham Salloum], [Raghad Alfaisal]),**  from it we can extract other related works where Chatbots were used in academic support, administration, and student engagement:

**Existing University Chatbot Solutions**

- **Unibuddy**: Connects prospective students with current students for insights on university life.
- **Ivy**: Assists with **administrative tasks** like financial aid and course registration.
- **HubSpot chatbots**: Provide instant responses to FAQs on university websites

## Methodology

- **Dataset**:
    - inspired from the paper/case study (**Building and Evaluating a Chatbot Using a University …)** we used an identical dataset from **kaagle** containing intents (e.g., course info, fees, hostel facilities), patterns (user queries), and responses.
    - https://www.kaggle.com/datasets/tusharpaul2001/university-chatbot-dataset
- Baseline:
    - After researching the best base-model that will suit our needs we filtered out the model that we will test out in our environment:
    | **Model** | **Size** | **Resource Usage** | **Performance** | **Status** |
| --- | --- | --- | --- | --- |
| **google/flan-t5-base** | 220M parameters | Low (lightweight) | Not highly performant | **Discarded** |
| **mistralai/Mistral-7B-V0.1** | 7B parameters | High (unable to load) | Strong performance (not tested) | **Discarded (resource limits)** |
| **gpt-neo-125M** | 125M parameters | Low (lightweight) | Weak performance | **Discarded** |
| **gpt-neo-1.3B** | 1.3B parameters | Medium (manageable) | Expected best fit (under testing) | **Currently Testing** |

- **Fine-tuning:**
    - now for the part that takes the longest (approximately 13 hours on current resources) ,
        - **Set Training Parameters**
            - Defines **batch size, number of epochs, save frequency**, and **logging settings** using `TrainingArguments`.
        - **Initialize Trainer**
            - The `Trainer` class is used to manage the training process, linking the **model, training arguments, and dataset**.
        - **Train the Model**
            - The `trainer.train()` command fine-tunes the model on the provided dataset.
        - **Save the Fine-Tuned Model**
            - The trained model and tokenizer are saved for future use.

```python
from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-gpt-neo")
tokenizer.save_pretrained("./fine-tuned-gpt-neo")
```
User interface with gradio and deployment into hugging face space :

```python
import gradio as gr

# Function to generate responses
def generate_response(user_input):
    try:
       ...
       "logic and parameters of repsonse plus error handelling"
       ...

# Gradio chat interface function
def respond(message, history):
  
# Create a Gradio ChatInterface
interface = gr.ChatInterface(
    fn=respond,
    title="University Customer Support Chatbot",
    description="Ask me anything about admissions, courses, or campus life!",
    examples=[
        "What are the admission requirements?",
       ...
    ]
)
"==> UI enhancing "

# Launch the interface
interface.launch()
```
- this code is responsible for the UI part of the project and directly gets “adopted” by the hugging face space once deployed
    - to deploy this we clone/init the hugging face repo, and add the code, then push the changes.
- The current  base model used in the space is **google/flan-t5-base** and not **gpt-neo-1.3B** since it’s not yet finished training for fine tuning purposes.

**Key Metrics from Training Output ( from google/flan-t5-base )**

| **Metric** | **Description** | **Business Impact** |
| --- | --- | --- |
| **Training Loss (29.35)** | Measures error during training (lower is better). | Helps gauge model accuracy. |
| **Global Steps (21)** | Total optimization steps taken. | Indicates training iterations. |
| **Train Runtime (470.33 sec)** | Total time taken for training. | Affects deployment feasibility. |
| **Samples per Second (0.319)** | Processing speed per sample. | Determines efficiency for scaling. |
| **Steps per Second (0.045)** | Model's update speed per second. | Helps in training time estimation. |
| **Total FLOPs (25.67T)** | Floating point operations during training. | Resource utilization indicator. |
| **Epochs (3.0)** | Number of training cycles. | Ensures sufficient learning. |

**User Experience & Engagement Metrics**

- **User Satisfaction** → Direct feedback on chatbot responses and usefulness.
- **Retention Rate** → Percentage of users who return for multiple interactions.
- **Drop-off Rate** → Percentage of users abandoning the chat mid-conversation.

**Business & Operational Efficiency**

- **Operational Cost Savings** → Reduction in support staff workload, leading to lower operational costs.
- **Adoption Rate** → Number of students using the chatbot vs. traditional support channels.

References:

https://www.researchgate.net/publication/382568739_Building_and_Evaluating_a_Chatbot_Using_a_University_FAQs_Dataset

https://www.researchgate.net/publication/388428246_A_review_of_university_chatbots_for_student_support_FAQs_and_beyond

https://www.kaggle.com/datasets/tusharpaul2001/university-chatbot-dataset

https://huggingface.co/EleutherAI/gpt-neo-1.3B/tree/main
