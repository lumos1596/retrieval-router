# Reasoning Router Casebook

Model: Qwen2.5-3B-Instruct, temperature=0. Comparisons on target datasets.

## tatdqa: P2 improved vs P0 (top 10)

**Q:** What was the percentage change in operating revenues between Three Months Ended  September and December?
- P0=TD(0.00) -> P2=MR(1.00) [d=+1.00]
- arms: MR=1.00 MD=1.00 TR=1.00 TD=0.00 BM=0.00
- P2: Task: fact lookup
Constraints: exact entities (Three Months Ended September, Three Months Ended December), number (percentage change)
Matching Need: lexical exact match
Comparison: For this query, both BM and MR could potentially work well due to the

**Q:** What is the cost of living indexation rate applicable to?
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=1.00 MD=1.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entities
Matching Need: lexical exact match
Comparison: For this query, BM (BM25) would likely perform better than TD (Text-Dense) because the question seeks an exact entity name (cost of living indexation rate), 

**Q:** What are the components of the impacts of cross-currency swap contracts in the table?
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=0.00 MD=1.00 TR=0.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: table
Matching Need: lexical exact match
Comparison: For this query, a lexical exact match is crucial since the question asks for specific components listed in a table. BM (BM25) is best suited for this task as it excel

**Q:** What was the total reported cases in FY 16?
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=1.00 MD=1.00 TR=0.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entities (total reported cases), year (FY 16)
Matching Need: lexical exact match
Comparison: TD (strong for semantic similarity but weak for exact keyword) vs. BM (strong for exact entity names and precise string 

**Q:** What is the percentage change in the end of period balance for Fulfillment Costs for the year ended December 31, 2019, compared to the begin
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=1.00 MD=1.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entities (Fulfillment Costs), years (2019), relation (percentage change)
Matching Need: lexical exact match
Comparison: This query requires finding specific financial data, which is best handled by BM (BM25) due t

**Q:** What method was used to value the acquired technology, in-process research and development assests? 
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=1.00 MD=1.00 TR=0.43 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entities
Matching Need: lexical exact match
Comparison: TD vs TR. TD is better for exact keyword matches, while TR might miss the exact phrase due to its focus on text density.
FINAL=BM

**Q:** What was the average income from operations for the cellular limited partnerships from 2017 to 2019?
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=1.00 MD=1.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entities (cellular limited partnerships), years (2017-2019), relation (average income from operations)
Matching Need: lexical exact match
Comparison: This query requires finding specific numerical data over a peri

**Q:** What is the average Buildings and Building Improvements value for 2018 and 2019?
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=0.43 MD=0.00 TR=0.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entities (Buildings and Building Improvements), years (2018, 2019)
Matching Need: lexical exact match
Comparison: This query requires finding specific numerical values for exact years. BM (BM25) is suitable due to

**Q:** What is the average revenue from the United States for the years 2017, 2018, and 2019, as shown in the revenue by geographic region table fo
- P0=MD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=0.39 MD=0.00 TR=0.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entities (United States, fitbit-inc), exact years (2017, 2018, 2019), relation (average revenue), multi-condition needs (years and geographic region)
Matching Need: lexical exact match
Comparison: TD vs TR. TD is 

**Q:** What was the change in Unearned subscription and SaaS revenue between 2019 and 2020?
- P0=TD(0.00) -> P2=MD(1.00) [d=+1.00]
- arms: MR=1.00 MD=1.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: fact lookup, numerical computation, constrained comparison, no explicit visual request, multi-condition needs
Matching Need: semantic recall, token-level fine-grained match
Comparison: TD vs TR
FINAL=MD

The question requires finding a specific

## tatdqa: P2 worsened vs P0 (top 10)

**Q:** What is the decrease in the value of Property, plant and equipment acquired by Cogeco Inc.'s subsidiary Atlantic Broadband from the prelimin
- P0=MD(1.00) -> P2=TD(0.00) [d=-1.00]
- arms: MR=1.00 MD=1.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: numerical computation
Constraints: exact entities (Cogeco Inc., Atlantic Broadband), years (2018, 2019), relation (decrease in property, plant and equipment value)
Matching Need: token-level fine-grained match
Comparison: TD vs. TR
TD is better

**Q:** What was the percentage change in Repairs and maintenance costs for Carpenter Technology Corp in 2019 compared to the previous year, as disc
- P0=MD(1.00) -> P2=BM(0.00) [d=-1.00]
- arms: MR=0.63 MD=1.00 TR=0.00 TD=1.00 BM=0.00
- P2: Task: fact lookup
Constraints: Carpenter Technology Corp, Repairs and maintenance costs, 2019, previous year
Matching Need: lexical exact match
Comparison: This query requires finding specific numerical data in a document. BM (BM25) is suitable for e

**Q:** What was the net income (loss) for Consolidated Communications Holdings Inc. in 2019, as reflected in the reconciliation of net income (loss
- P0=MD(1.00) -> P2=BM(0.00) [d=-1.00]
- arms: MR=1.00 MD=1.00 TR=0.43 TD=1.00 BM=0.00
- P2: Task: fact lookup
Constraints: Consolidated Communications Holdings Inc., 2019, net income (loss), Adjusted EBITDA
Matching Need: lexical exact match
Comparison: For this query, BM (BM25) is likely the best choice because it excels at finding exact e

**Q:** In the additional information relating to the total assets of American Tower Corporation's operating segments for the years ended December 3
- P0=MD(1.00) -> P2=BM(0.00) [d=-1.00]
- arms: MR=0.63 MD=1.00 TR=0.50 TD=1.00 BM=0.00
- P2: Task: fact lookup
Constraints: exact entities (American Tower Corporation), exact years (years ended December 31), number (300 million)
Matching Need: lexical exact match
Comparison: This query requires finding specific numerical data in a report abo

**Q:** Where does the company include the non-current portion of capitalized commission costs and upfront payments made to customers in their conso
- P0=MD(1.00) -> P2=TD(0.00) [d=-1.00]
- arms: MR=1.00 MD=1.00 TR=0.43 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: company name, specific financial term
Matching Need: semantic recall
Comparison: TD (strong for semantic similarity) vs. TR (fine-grained multi-condition matching); TD is better suited for understanding the financial co

**Q:** What is the maturity of long-term debt for the year immediately following 2020, as outlined in the maturities of long-term debt section of C
- P0=MD(1.00) -> P2=BM(0.00) [d=-1.00]
- arms: MR=0.63 MD=1.00 TR=1.00 TD=1.00 BM=0.00
- P2: Task: fact lookup
Constraints: Cubic Corporation, 2019 financial report, long-term debt maturity, year immediately following 2020
Matching Need: lexical exact match
Comparison: This query requires an exact match of the phrase "maturity of long-term d

**Q:** What is the difference in operating income between the years 2015 and 2016 as reported in ManTech International Corp's selected financial da
- P0=MD(1.00) -> P2=BM(0.00) [d=-1.00]
- arms: MR=0.50 MD=1.00 TR=0.00 TD=0.00 BM=0.00
- P2: Task: fact lookup
Constraints: exact entities (ManTech International Corp), years (2015, 2016, 2019), relation (operating income difference)
Matching Need: lexical exact match
Comparison: For this query, BM (BM25) would likely perform better due to i

**Q:** What was the effect of the company's indirect investment in The Dun and Bradstreet Corporation?
- P0=MD(1.00) -> P2=TD(0.00) [d=-1.00]
- arms: MR=0.39 MD=1.00 TR=0.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: company name, corporation name, year
Matching Need: semantic recall
Comparison: TD (strong for semantic similarity) vs. TR (good for relevance discrimination within text)
FINAL=TD

**Q:** What was the value of Other taxes receivable for coherent-inc in the 2019 fiscal year, as reported in the breakdown of prepaid expenses and 
- P0=MD(1.00) -> P2=BM(0.00) [d=-1.00]
- arms: MR=1.00 MD=1.00 TR=0.00 TD=1.00 BM=0.00
- P2: Task: fact lookup
Constraints: coherent-inc, Other taxes receivable, 2019 fiscal year, prepaid expenses and other assets
Matching Need: lexical exact match
Comparison: This query requires finding a specific piece of financial data in a report. BM (BM

**Q:** What was the total amount of Goodwill for black-knight-financial-services-inc as of December 31, 2019?
- P0=MD(1.00) -> P2=BM(0.00) [d=-1.00]
- arms: MR=1.00 MD=1.00 TR=0.00 TD=0.50 BM=0.00
- P2: Task: fact lookup
Constraints: exact entity (black-knight-financial-services-inc), year (2019), relation (Goodwill), date (December 31)
Matching Need: lexical exact match
Comparison: This query requires finding a specific piece of financial data, whi

## wiki-ss: P2 improved vs P0 (top 10)

**Q:** what kind of creature is uniqua from the backyardigans
- P0=TD(0.00) -> P2=MR(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=0.63 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entity name: Uniqua
Matching Need: lexical exact match
Comparison: TD vs MR
TD is better suited for semantic recall, while MR excels in handling explicit visual requests. For this query, MR might be more effective

**Q:** who played bass on walk on the wild side
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=0.00 MD=0.00 TR=0.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entities (names of musicians)
Matching Need: lexical exact match
Comparison: TD vs TR
TD is better suited for semantic similarity, while TR excels in fine-grained multi-condition matching. For this query, finding 

**Q:** where was angus thongs and perfect snogging filmed
- P0=MD(0.00) -> P2=TD(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=0.00 TD=1.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact location
Matching Need: semantic recall
Comparison: TD (strong for semantic similarity) vs. TR (good for relevance discrimination within text)
FINAL=TD

**Q:** who performed with beyonce at the super bowl
- P0=TD(0.00) -> P2=TR(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=0.00
- P2: Task: explicit visual request
Constraints: exact entities
Matching Need: semantic recall
Comparison: TD vs TR
TD is better for semantic similarity, while TR excels in fine-grained multi-condition matching. For this query about performers at the Super

**Q:** where is the most oil produced in the us
- P0=MD(0.00) -> P2=TR(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=0.63
- P2: Task: constrained comparison
Constraints: exact entities
Matching Need: token-level fine-grained match
Comparison: TD vs TR
TD is better for semantic similarity and paraphrased questions, while TR excels in fine-grained multi-condition matching. For 

**Q:** who sings the song everybody must get stoned
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: explicit visual request
Constraints: none
Matching Need: lexical exact match
Comparison: TD vs TR. TD is better for exact keyword matches, while TR excels in relevance discrimination within text.
FINAL: BM

**Q:** who played william harris on law and order svu
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=1.00 MD=1.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entity name: William Harris
Matching Need: lexical exact match
Comparison: TD vs TR
TD is better suited for exact entity names as it uses single-vector dense retrieval which can match the query precisely. TR might

**Q:** when did west brom fc change its name
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=0.50 MD=0.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entity name (West Brom FC)
Matching Need: lexical exact match
Comparison: TD (strong for exact keyword matches) vs. BM (weak for semantic paraphrase but strong for exact entity names)
FINAL=BM

**Q:** who played will scarlett in robin hood prince of thieves
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entity name (Will Scarlett)
Matching Need: lexical exact match
Comparison: TD (strong for exact keyword) vs. BM (weak for semantic paraphrase but strong for exact keyword)
FINAL=BM

**Q:** what do you call a game of bowling
- P0=TD(0.00) -> P2=BM(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: no specific exact entities needed
Matching Need: lexical exact match
Comparison: This query requires a lexical exact match. BM (BM25) is suitable here because it excels at finding exact entity names, which in this case 

## wiki-ss: P2 worsened vs P0 (top 10)

**Q:** who wrote the song love on a two way street
- P0=TD(1.00) -> P2=BM(0.00) [d=-1.00]
- arms: MR=0.00 MD=0.00 TR=0.00 TD=1.00 BM=0.00
- P2: Task: fact lookup
Constraints: exact entity name (song writer)
Matching Need: lexical exact match
Comparison: TD (strong for exact keyword) vs. BM (weak for semantic paraphrase)
FINAL=BM

**Q:** who owns the lord of the rings franchise
- P0=TD(1.00) -> P2=MR(0.00) [d=-1.00]
- arms: MR=0.00 MD=1.00 TR=0.00 TD=1.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entities
Matching Need: lexical exact match
Comparison: TD vs MR
TD is better suited for finding exact entity names in text-only pages, while MR excels at handling pages with visual elements like figures or tables

**Q:** what does ncd stand for in medical terms
- P0=TD(1.00) -> P2=BM(0.00) [d=-1.00]
- arms: MR=0.00 MD=1.00 TR=1.00 TD=1.00 BM=0.00
- P2: Task: fact lookup
Constraints: exact entities
Matching Need: lexical exact match
Comparison: TD vs BM
TD is better suited for semantic similarity and paraphrasing, which might help in finding "ncd" in medical terms. BM, however, excels at exact keywo

**Q:** where is the main office of the ncaa
- P0=TD(1.00) -> P2=BM(0.00) [d=-1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=1.00 BM=0.00
- P2: Task: fact lookup
Constraints: exact entity (ncaa)
Matching Need: lexical exact match
Comparison: TD vs TR
TD is better suited for exact entity names, while TR excels in fine-grained multi-condition matching. For the query "where is the main office o

**Q:** who played wonder woman 's mother in new movie
- P0=TD(1.00) -> P2=MR(0.00) [d=-1.00]
- arms: MR=0.00 MD=0.39 TR=0.00 TD=1.00 BM=0.43
- P2: Task: explicit visual request
Constraints: exact entity (Wonder Woman's mother)
Matching Need: lexical exact match
Comparison: TD vs TR
TD is better suited for semantic similarity, while TR excels in fine-grained multi-condition matching. However, gi

**Q:** who plays thor 's father in the movie
- P0=TD(1.00) -> P2=TR(0.00) [d=-1.00]
- arms: MR=0.00 MD=1.00 TR=0.00 TD=1.00 BM=1.00
- P2: Task: explicit visual request
Constraints: exact entities
Matching Need: lexical exact match
Comparison: TD vs TR
TD is better suited for semantic similarity and paraphrased questions, while TR excels in fine-grained multi-condition matching. For the

**Q:** who wrote stay with me by the faces
- P0=BM(1.00) -> P2=TD(0.00) [d=-1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: exact entity names (The Faces)
Matching Need: lexical exact match
Comparison: TD vs TR
TD is better suited for exact entity names as it uses single-vector dense retrieval over text, making it strong for semantic similar

**Q:** when did gulf oil go out of business
- P0=BM(1.00) -> P2=TD(0.00) [d=-1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: fact lookup
Constraints: years
Matching Need: semantic recall
Comparison: TD (strong for semantic similarity) vs. TR (strong for relevance discrimination within text)
FINAL=TD

**Q:** when are spermatozoa transported from the vas deferens to the abdominal urethra
- P0=MD(1.00) -> P2=TR(0.00) [d=-1.00]
- arms: MR=0.00 MD=1.00 TR=0.00 TD=1.00 BM=1.00
- P2: Task: fact lookup
Constraints: years, number of conditions
Matching Need: semantic recall
Comparison: TD vs TR
TD is better for semantic similarity and paraphrased questions, while TR excels in fine-grained multi-condition matching. However, given th

**Q:** who founded four mathas in the four corners of india
- P0=TD(1.00) -> P2=BM(0.00) [d=-1.00]
- arms: MR=0.63 MD=1.00 TR=0.63 TD=1.00 BM=0.00
- P2: Task: fact lookup
Constraints: exact entities (Four Mathas, Four corners of India)
Matching Need: lexical exact match
Comparison: TD vs TR
TD is better suited for exact entity names like "Four Mathas" and "Four corners of India," while TR can provide

## arxivqa: P2 improved vs P0 (top 10)

**Q:** What can be inferred about the data points with error bars compared to those without?
- P0=TD(0.00) -> P2=TR(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=0.00
- P2: Task: constrained comparison  
Constraints: relations  
Matching Need: semantic recall  
Comparison: TD vs TR  
TD is better for semantic similarity, while TR excels in fine-grained multi-condition matching. For this query about data points with erro

**Q:** Based on the diagram in part (b), what is the role of the matrices labeled \( T_i \) and \( T_{i+1} \)?
- P0=MD(0.00) -> P2=TR(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: explicit visual request  
Constraints: none  
Matching Need: page visual representation  
Comparison: TD vs TR  
TD is better at semantic similarity, but TR can provide more fine-grained relevance discrimination within text. For this query, TR 

**Q:** What does the parameter 'D' represent in panels (c) and (f) of the figure?
- P0=TD(0.00) -> P2=MR(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=0.50 TD=0.00 BM=0.00
- P2: Task: explicit visual request  
Constraints: none  
Matching Need: page visual representation  
Comparison: MR vs. MD  
FINAL=MR

**Q:** What physical phenomenon is likely being modeled in these figures?
- P0=MD(0.00) -> P2=MR(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=0.00
- P2: Task: explicit visual request
Constraints: none
Matching Need: page visual representation
Comparison: This query requires a strong visual understanding of the figures. A retriever like MR (MM-Rerank) would be best suited for this task as it combines 

**Q:** What statistical distribution is most likely being represented by the polar histogram in figure (b)?
- P0=TD(0.00) -> P2=MD(1.00) [d=+1.00]
- arms: MR=0.00 MD=1.00 TR=0.00 TD=0.00 BM=1.00
- P2: Task: explicit visual request
Constraints: none
Matching Need: explicit visual request
Comparison: TD (strong for semantic similarity) vs. TR (strong for fine-grained multi-condition matching)
FINAL=MD

**Q:** Compared to the left graph, what does the right graph indicate about the rate of improvement in Jac.C with increasing initializations?
- P0=TR(0.00) -> P2=MD(1.00) [d=+1.00]
- arms: MR=0.00 MD=1.00 TR=0.00 TD=1.00 BM=0.43
- P2: Task: explicit visual request
Matching Need: page visual representation
Comparison: TD vs TR
FINAL=MD

**Q:** What is the significance of the different shapes (square and arrow) used for \(\bar{x}_B\) and \(\bar{x}_{B,i}\)?
- P0=TD(0.00) -> P2=MR(1.00) [d=+1.00]
- arms: MR=1.00 MD=1.00 TR=1.00 TD=0.00 BM=0.00
- P2: Task: explicit visual request
Constraints: shapes, \(\bar{x}_B\), \(\bar{x}_{B,i}\)
Matching Need: page visual representation
Comparison: TD vs TR
TD is better at semantic similarity and paraphrasing, but lacks visual information needed for identifyi

**Q:** What does the dashed line in the contour plots signify?
- P0=TD(0.00) -> P2=TR(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=0.00
- P2: Task: explicit visual request
Constraints: none
Matching Need: page visual representation
Comparison: TD (Text-Dense) vs TR (Text-Rerank)
TD is better at semantic similarity and paraphrasing, which might not directly address the visual query. TR, how

**Q:** What does the image labeled "1 - MR - MT" represent when comparing it to the "MR" and "MT" brain images?
- P0=MD(0.00) -> P2=TR(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=0.00
- P2: Task: explicit visual request
Constraints: exact entities
Matching Need: page visual representation
Comparison: TD vs TR
TD is better at semantic similarity and paraphrased questions, while TR excels in fine-grained multi-condition matching. For this

**Q:** What does the infrared (IR) laser primarily interact with in this setup?
- P0=MD(0.00) -> P2=MR(1.00) [d=+1.00]
- arms: MR=1.00 MD=0.00 TR=0.00 TD=0.00 BM=0.00
- P2: Task: explicit visual request
Constraints: none
Matching Need: page visual representation
Comparison: TD (Text-Dense) is better at handling visual requests due to its single-vector dense retrieval over text, while TR (Text-Rerank) might miss the visu

## arxivqa: P2 worsened vs P0 (top 10)

**Q:** What does the red arrow in Figure b most likely indicate?
- P0=MR(1.00) -> P2=MD(0.00) [d=-1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=0.00
- P2: Task: explicit visual request
Constraints: none
Matching Need: page visual representation
Comparison: TD (strong for semantic similarity) vs. MD (strong for visual layout/content)
FINAL=MD

**Q:** Based on the graph, which of the following best describes the behavior of the output power between 29 W and 30 W of absorbed pump power?
- P0=MD(1.00) -> P2=TR(0.00) [d=-1.00]
- arms: MR=1.00 MD=1.00 TR=0.00 TD=0.39 BM=1.00
- P2: Task: explicit visual request
Constraints: graph, output power, absorbed pump power
Matching Need: explicit visual request
Comparison: TD (strong for semantic similarity) vs. TR (strong for fine-grained multi-condition matching)
FINAL=TR

**Q:** What trend is observed for the RBM method as the problem size N increases in Figure a?
- P0=TD(1.00) -> P2=TR(0.00) [d=-1.00]
- arms: MR=0.00 MD=1.00 TR=0.00 TD=1.00 BM=1.00
- P2: Task: explicit visual request  
Constraints: none  
Matching Need: page visual representation  
Comparison: TD vs TR  
TD is better at semantic similarity, but TR can provide finer-grained relevance discrimination within text. For a visual trend anal

**Q:** What effect does increasing the force constant \( \tilde{f} \) have on the arrangement of particles as depicted in the figures?
- P0=MD(1.00) -> P2=TR(0.00) [d=-1.00]
- arms: MR=0.00 MD=1.00 TR=0.00 TD=0.00 BM=1.00
- P2: Task: explicit visual request
Constraints: figures
Matching Need: page visual representation
Comparison: TD vs TR
TD is better at semantic similarity and paraphrased questions, while TR excels in fine-grained multi-condition matching and relevance di

**Q:** In figures (d), (e), and (f), what distribution characteristic is depicted by the blue shaded area under the curves?
- P0=MD(1.00) -> P2=TR(0.00) [d=-1.00]
- arms: MR=0.63 MD=1.00 TR=0.00 TD=0.00 BM=0.00
- P2: Task: explicit visual request
Constraints: exact entities (figures (d), (e), (f))
Matching Need: page visual representation
Comparison: TD vs TR
TD is better at semantic similarity and paraphrased questions, while TR excels in fine-grained multi-cond

**Q:** Which method demonstrates the greatest variability in performance with changes in the simulation budget in the Spatial SIR model?
- P0=MR(1.00) -> P2=TD(0.00) [d=-1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: constrained comparison
Constraints: years
Matching Need: semantic recall
Comparison: TD vs TR
TD excels in semantic similarity, while TR can refine relevance within text but lacks visual interaction. For a query about simulation budget variabil

**Q:** Based on panel B, what can be inferred about the trend of coins over time?
- P0=MD(1.00) -> P2=TR(0.00) [d=-1.00]
- arms: MR=1.00 MD=1.00 TR=0.00 TD=0.43 BM=0.63
- P2: Task: explicit visual request
Constraints: panel B
Matching Need: explicit visual request
Comparison: TD (strong for semantic similarity) vs TR (strong for fine-grained multi-condition matching)
FINAL=TR

**Q:** Based on the detail provided in the zoomed-in section of the difference image, what can be inferred about the nature of the movement between
- P0=MD(1.00) -> P2=TR(0.00) [d=-1.00]
- arms: MR=0.00 MD=1.00 TR=0.00 TD=0.00 BM=0.00
- P2: Task: explicit visual request
Constraints: difference image, zoomed-in section, \(I(x, y, t)\), \(I(x, y, t + \Delta t)\)
Matching Need: page visual representation
Comparison: TD vs TR
TD is better for semantic similarity and paraphrased questions, w

**Q:** According to panel B, how do 'Start cost' and 'End cost' compare as the mixing \( x \) approaches 1?
- P0=TR(1.00) -> P2=TD(0.00) [d=-1.00]
- arms: MR=1.00 MD=1.00 TR=1.00 TD=0.00 BM=1.00
- P2: Task: constrained comparison
Constraints: exact entities: panel B, 'Start cost', 'End cost', mixing \( x \)
Matching Need: semantic recall
Comparison: TD vs TR
TD is strong for semantic similarity and paraphrased questions, making it suitable for und

**Q:** What can be inferred about the time it takes for the system to reach a steady state in the different scenarios presented in the figure?
- P0=TR(1.00) -> P2=MD(0.00) [d=-1.00]
- arms: MR=1.00 MD=0.00 TR=1.00 TD=0.00 BM=0.00
- P2: Task: explicit visual request  
Matching Need: page visual representation  
Comparison: MD vs TR  
FINAL=MD
