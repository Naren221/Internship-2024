<h1 align = 'center'> Cognitive Assessment of Student Questions: A SOLO Taxonomy Approach" </h1> 

## Problem Context
The primary objective of this project is to evaluate the `quality of questions` posed by students, with the underlying assumption that the quality of their questions reflects their level of understanding of the course concepts. Additionally, we aim to `automate the tagging` of these questions with relevant concepts, such as Python, data structures, Apache Spark, etc., to enhance categorization and retrieval. These tags will not only aid in organizing and searching questions but will also be integral to our primary goal of assessing the students' cognitive levels and understanding of the material.

To perform this analysis, we require a dataset containing student-asked questions, along with their associated tags. Such data can be sourced from `Stack Overflow (SOF)`, a platform where users ask and answer technical questions. It is also crucial to differentiate between individual users on Stack Overflow, as this allows us to analyze the data on a **per-user basis**, giving insight into each user's understanding and progress. Furthermore, understanding the trends in the types of questions asked can provide valuable feedback on the areas where students are struggling or excelling, helping educators to tailor their teaching strategies accordingly. The automated tagging and subsequent analysis will enable a scalable and efficient way to monitor and improve **educational outcomes**.

## Problem Statements/ Questions being investigated
1. What are the trends seen in questions posted in SOF for the concepts related to broad area of DataStructures and Algorithms ?

2. How can we classify questions posted in SOF/QP sites so that it conveys the complexitites associated with framing of the question and the cognitive levels of the learner ?

3. What methods can be used to classify the questions posted and how effective are they ?


## Solution Approach

### Approach for PS1: Trends in DataStructures and Algorithms Questions

To investigate the trends in questions related to `Data Structures and Algorithms (DSA)` on Stack Overflow (SOF), we employed the following approach:

- **Data Extraction and Preprocessing**:

    **Data Collection**: We extracted question data from Stack Overflow, focusing on tags and content related to Data Structures and Algorithms. The dataset includes question IDs, titles, tags, and other metadata.

    **Data Cleaning**: Tags associated with each question were cleaned and converted from string representations into lists for easier manipulation. We ensured that all columns were standardized to lowercase to avoid inconsistencies.

- **Exploratory Data Analysis (EDA)**:
    
    **Tag Analysis**: We performed a frequency analysis on tags to identify the most common tags related to DSA. This involved counting occurrences of each tag and plotting the top tags to visualize their distribution.

    **Trend Analysis**: We examined temporal trends in question posting, including the distribution of DSA-related questions over time. This helps in understanding whether there are seasonal or emerging trends in DSA topics.

- **Tag Frequency Analysis**:
    
    **Top Tags Identification**: We identified the top tags associated with DSA questions and visualized their frequencies using bar plots. This provided insight into the most frequently discussed topics within DSA.

    **Tag Distribution**: We analyzed the cumulative distribution of tags to understand how many tags cover a significant proportion of the total occurrences, revealing the concentration of discussions around a few popular tags versus a broader set of tags.

- **Question Length and Complexity**:
    
    **Length Analysis**: We assessed the length of questions to identify any trends related to the complexity of questions. This involved analyzing the minimum, maximum, and average length of question titles.
    
    **Tag Count per Question**: We calculated the number of tags per question to understand how detailed or focused the questions are. This provided insights into the level of specificity in DSA questions.

- **Visualization**:

    **Bar Plots**: Visualizations were created to represent the distribution of top tags and the number of questions associated with each tag. This helped in easily identifying popular and less common topics within DSA.

    **Time Series Analysis**: We used time series plots to visualize the trends in question posting over time, highlighting any significant changes or patterns.

- **Summary and Insights**:

    **Trend Summary**: We summarized the key findings from the tag and question length analyses. This included identifying which DSA topics are most frequently discussed and whether there are any notable trends in the types of questions asked.
    
    **Educational Implications**: The insights from this analysis can help educators understand the areas where students are most engaged or struggling, allowing them to tailor their instruction and support accordingly.

By following this approach, we were able to provide a comprehensive analysis of the trends in DSA-related questions on Stack Overflow


### Approach for PS2: Framework for Classification of Questions

To effectively classify questions based on their complexity and cognitive level, we need a robust method that can categorize the depth of understanding reflected in the questions. Our goal is to classify questions in a way that conveys the complexities associated with framing the question and the cognitive levels of the learner. This can be achieved using a framework called SOLO Taxonomy, which offers a structured approach to evaluating and categorizing the depth of student responses and questions.

**Why SOLO Taxonomy?**

SOLO Taxonomy (Structure of Observed Learning Outcomes) is particularly well-suited for this task due to its structured and hierarchical approach to evaluating learning outcomes. Developed by John Biggs and Kevin Collis, SOLO Taxonomy provides a clear framework for assessing the quality of understanding by categorizing responses into different levels of complexity. Here’s why SOLO Taxonomy is advantageous for our problem:

* **Hierarchical Structure**: SOLO Taxonomy organizes learning outcomes into a hierarchy of levels, from basic to advanced understanding. This aligns well with our need to classify questions based on the depth of cognitive engagement. By using SOLO Taxonomy, we can systematically categorize questions according to their complexity, ranging from simple, single-aspect questions to complex, multi-faceted queries.

* **Focus on Cognitive Complexity**: The taxonomy emphasizes the demands of responses, helping to distinguish between questions that reflect a basic understanding of a topic and those that demonstrate a deeper, more integrated grasp of concepts. For technical subjects like Data Structures and Algorithms, where understanding can vary significantly in depth, this feature is crucial.

* **Applicability Across Domains**: SOLO Taxonomy is versatile and can be applied to various domains and types of questions. In the context of our project, it provides a flexible framework for evaluating questions related to complex technical subjects, ensuring that the classification is relevant and accurate.

**SOLO Taxonomy Levels**:

* **Prestructural**: The response shows a lack of understanding or irrelevant information.

* **Unistructural**:  The response demonstrates a basic understanding of a single aspect of the topic.

* **Multistructural**: The response covers multiple aspects of the topic but lacks integration.

* **Relational**: The response integrates multiple aspects of the topic, showing a deeper understanding.

* **Extended Abstract**: The response demonstrates a high level of understanding by applying and extending the concepts in new or abstract ways.

By adopting SOLO Taxonomy, we can effectively classify questions based on their cognitive complexity and provide insights into the levels of understanding reflected in student queries. This structured approach will aid in evaluating the depth of engagement and support the development of educational strategies tailored to varying levels of comprehension.

### Approach for PS3: Automating tagging of questions and building an algorithm for classifying questions.

1. **Automating tagging of questions**:
    
    **Algorithms and Techniques**:
    - I have used Scikit-learn to extract useful features (tf-idf) from the data-frame.
    - As the problem is multilabel classification problem, I have used OneVsRestClassifier from scikit-learn to classify one tag versus all the tags at a time.
    - The model is trained by using SGDClassifier as the estimator algorithm.

    **Methodology**:
    I have organized all the crucial steps in a jupyter notebook
    - Exploratory Data Analysis
    - Text preprocessing
    - Feature Engineering
    - Model Training
    
    **Text Prer-processing**:
    - Firstly, Pandas is used to read the two CSV files. The two data frames are `questions_df`, which contains two columns: one for questions and a unique id associated with each question. The `tags_df` data frame also contains two columns: id, which represents the unique identifier for a question, and tag, which lists the relevant tags describing the question's topic. 
    - After some exploratory analysis, the two dataframes are joined so that we have all tags in the same record. 
    - The dataset is then further standardised by converting text to lower cases, removing stopwords and tokenizing the text using NLTK.
    - I have also kept a list of popular tags with symbols to work around how NLTK tokenize words, as they are not considered legal words in NLTK.

    **Feature Engineering**:
    - TfidfVectorizer from scikit-learn is used to extract term frequency-inverse document frequency (tf-idf) as features from title and body. Tf-idf is a numeric statistic to reflect how important a word is to a document in a collection, which is very useful in document classification.
    - I have included 10,000 most common words from question/title s features.
    - The labels(ie. tags) are then transformed from the intuitive format (eg. [‘c#’, ‘.net’, ‘ms-word’]) into an array of 1301 binary classes so that it can be fit for model training.

    **Model Training**:
    - As this is a multilabel classification, I used OneVsRestClassifier from scikit-learn to fit one classifier per class (i.e. 1350 classifiers will be trained in total, where each class is fitted against all other classes). 
    
    - I used SGDClassifier as the underlaying algorithm for classification.
    
    **Model Evaluation**:
    - The final model is evaluated using hamming loss, which is defined as the fraction of labels that are incorrectly predicted. I have also included other metrics such as precision, recall and F score so that we can have a better understanding on the model performance.
    
    - **All 4000 tags**
    <p align="center">
    <img src="Report Images\image.png" alt="Multi-Head model" />
    </p>

    - Just by a quick glance, we can see that the results for some tags are quite good. The overall results are not as good as expected, which is because there are a lot of tags that results in no precision at all.
    - The reason that a lot of tags have no predictions, is because:
        - they have too few examples (True count) in the datset
        - they are being tagged for the main topic instead of the    subtopics.
    - We can filter them out to see how the model actually preforms for the major tags.
    
    - **Top ten tags**
    <p align="center">
    <img src="Report Images\image-1.png"/>
    </p>


    - The metrics for the top 10 tags are actually pretty good . It’s partly because they have a lot of samples (true count) in the dataset.

    **Justification**
    - We cannot get a good picture on how the models perform overall, as a lot of classes have too few examples in the dataset, resulting in no predictions, even though we have limited it to the top 1350 tags.
    - In retrospect, we can further limit the dataset to top 1000 tags before training the model, or perform some graph analysis to have a better understanding among topics and subtopics within the tags, and apply filters accordingly. The problem may then be solved by other approaches, such as Hierarchical Classification.
    - To work around for now, we can filter out the classes with no prediction at all and see how the models perform.

    
        <p align="center">
        <img src="Report Images\image-4.png" alt="Result" />
        </p>


    **Hybrid Model**
    - The hybrid model predicts relevant tags for a given question using a combination of machine learning  and keyword matching. It preprocesses the question by converting it to lowercase, tokenizing it, and removing non-alphabetic characters. The pre-processed text is then transformed using a TF-IDF vectorizer and passed to a pre-trained SGD classifier to get initial tag predictions. Simultaneously, the function performs keyword matching by identifying and extracting predefined tags from the question using regular expressions. 
    - The combined tags from both methods are filtered to remove substrings and duplicates, providing a comprehensive set of relevant tags.

2. **Algorithm for SOLO Classification of quesitons**:

    I created an algorithm for our classification task utilizing the SOLO Taxonomy.

    **Key Concepts**:

    `Relation keywords`:
        This is a list of tuples representing significant relationships between keywords in the question.
        These relationshps are extracted based on  dependecy parsing, focusing on subjects,objects and other key grammatical roles.

    `max_distance`: 
        This is the maximum distance between any two keyword embeddings in the question. 
        A lower maximum distance indicates that even the most distant concepts in the question are still relatively close, which  may suggest a very tight integration of ideas.

    `avg_distance`: 
        This is the average of the pairwise distances (cosine similarities) between the embeddings of keywords in the question. 
        A lower average distance indicates that the concepts are semantically closer to each other, suggesting better integration.

    **Thresholds**

    `Strong threshold`:
        Determines a threshold for strong keyword relationships.
        Helps classify questions as "Relational" or "Extended Abstract".
        Example: `strong_threshold = 0.2` indicates strong relationships below this distance.

    `Weak threshold` :
        Sets a boundary for weak keyword relationships.
        Influences classification towards "Multi-structural" if exceeded.
        Example: `weak_threshold = 0.4` identifies weak relationships above this distance.

    **Classification Logic** :

    We classify a question into different levels of the SOLO taxonomy based on keywords extracted from the question and their relationships. The SOLO (Structure of Observed Learning Outcome) taxonomy levels are: Pre-structural, Uni-structural, Multi-structural, Relational, and Extended Abstract. 

    **Inputs**
    - question: The question text to be classified.
    - Strong threshold : A threshold for determining strong relationships (default is 0.18).
    - Weak threshold : A threshold for determining weak relationships (default is 0.2325).

    **Classification Steps** :

    1. **Keyword Extraction** :
        
        Extract relevant keywords from the question, which provide insights into various aspects of the course the student is inquiring about.

    2. **Initial Checks**
        - If no keywords are found in the question, then return `Pre-structural`.
        - If only one keyword in found
            - Calculate the embedding for the entire question and the  keyword.
            - Then calculate the cosine distance between the question embedding and keyword embedding.
            - If the minimum distance is greater than 0.4(or some fixed threshold), return `Pre-structural`.
            - Otherwise, it check if any of the extended abstract keywords are in the question. If so, it return "Extended Abstract". If not, return `Uni-structural`.

    3. **Distance Calculation**.
        - Calculate the distance matrix for the keywords.
        - Then find the maximum distance and the average distance of the upper triangle of the distance matrix.

    4. **Syntactic Structure Analysis(Dependency parsing)**.
        - Analyzes the syntactic structure of the question to find relationships between words.
        - Identify keyword relationships within these syntactic structures.

    5. **Extended abstract keywords**
        - Create a list of keywords that are commonly found in extended abstract questions.

    6. **If no relationships between keywords are found**.
        - If no relationships between keywords are found and the maximum distance between keywords is greater than or equal to the weak threshold, return `Multi-structural`.
        - If there are only two keywords, Calculate the embedding for the entire question and each keyword.
        - Then calculate the minimum cosine distance between the question embedding and each keyword embedding.
        - If the minimum distance is greater than 0.4(or some predefined fixed threshold), return `Pre-structural`.
        - Otherwise, check if any of the extended abstract keywords are in the question. If so, return `Extended Abstract`.  If not, return `Uni-structural`.

    7. **If relationships between keywords are found**:
        - If the average distance is less than or equal to the strong threshold, return `Relational`.
        - If the maximum distance is less than or equal to the strong threshold, return `Extended Abstract`. Otherwise, return `Relational`.

    Overall, the Algorithm follows a hierarchical decision-making process to classify a question into one of the SOLO taxonomy levels based on:
    - The presence and number of keywords.
    - The distances between keyword embeddings.
    - The syntactic structure of the question and relationships between keywords.
    - Specific thresholds for strong and weak relationships.
        
## Results.

### For PS1: Trends in DataStructures and Algorithms Questions.

**Top 10 tags**.
    <p align="center">
    <img src="Report Images\image-5.png" alt="Top 10 tags" />
    </p>



**Top 50 tags**.
    <p align="center">
    <img src="Report Images\image-6.png" alt="Top 50 tags" />
    </p>

**Analysis of Tags and Question Length**:

- The distribution of the number of questions per tag exhibits a long-tail pattern. To enhance the efficiency of model training while preserving accuracy, we can limit the number of tags included in the dataset.

- We determined that 1,301 tags cover at least 90% of the total occurrences. Therefore, we will use the top 1,301 tags for building the classification model.

- Each question is tagged with a minimum of 1 tag and a maximum of 5 tags, with an average of 3.46 tags per question.

- The length of questions varies, with a minimum length of 15 characters and a maximum length of 150 characters.

### For PS3 
* I have used the algorithm to Classify some of the questions on Data structures and Algorithms i was able to find on Stack overflow. Here are the results .


<p align="center">
    <img src="Report Images\image-3.png" alt="final Result" />
    </p>

### Scope of Improvement
* After applying the model to over 100 questions, the initial results suggest that it is functioning effectively. However, a comprehensive evaluation of the algorithm is imperative to confirm its reliability. One of the primary challenges we face is the lack of an existing labeled dataset that classifies questions according to SOLO Taxonomy. To address this gap, a set of questions will need to be manually classified by a subject matter expert. This process will provide the necessary labeled data to rigorously evaluate and fine-tune the model.

* Currently, the model's classifications may be influenced by my own interpretation, which could introduce bias. To overcome this, incorporating the expertise of a subject matter expert is essential. This will not only help validate the model's accuracy but also ensure that it aligns with the pedagogical principles underlying SOLO Taxonomy. Additionally, once a validated dataset is available, further testing and refinement can be conducted to enhance the model's generalizability and effectiveness across a broader range of questions. This step is crucial for achieving a more objective and robust classification system that accurately reflects the cognitive levels and complexities intended by the taxonomy.

### Insights or New Learning
This project has provided valuable insights into how automated question classification can be used to evaluate students on an individual basis. By applying frameworks like SOLO Taxonomy, we can categorize questions according to their complexity and cognitive demand, offering a deeper understanding of each student's grasp of the material. This approach enables educators to assess not just the quantity but the quality of student interactions, identifying areas where individual students may need more support or advanced challenges. Ultimately, this method can tailor educational experiences to better match each student's unique learning needs.

### References or external links
- https://www.inspiringinquiry.com/learningteaching/toolsstrategies/solo-taxonomy

#### Github
https://github.com/Naren221/SHARP

#### Research Papers
* [https://cs229.stanford.edu/proj2014/Mihail%20Eric,%20Ana%20Klimovic,%20Victor%20Zhong,MLNLP-Autonomous%20Tagging%20Of%20Stack%20Overflow%20Posts.pdf]


