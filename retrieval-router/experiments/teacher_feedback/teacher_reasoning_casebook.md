# Teacher Reasoning Casebook

## Purpose

Contrast **evidence-grounded** explanations (T2, with real Top-5 ranking + GT flags) 
against **post-hoc rationalization** (T1, nDCG-only, capability stereotypes).

Sample: 120 source-domain queries (finqa/vqaonbd), Qwen2.5-3B-Instruct, temp=0.

All 120 queries have real Top-5 retrieval ranking evidence (dense Top-5 with GT flags; 13 also have MR rerank, 2 have TR rerank).


---

## 1. Credible (Evidence-Grounded) Explanations

These are T2 outputs where the model was given real Top-5 dense/rerank rankings with GT flags.


### C1 | tie | MR=1.000 vs MD=1.000

**Query:** What percentage of American Airlines' total planes, as of December 31, 2017, had an average seating capacity of fewer than 100 passengers?

**Evidence arms:** MULTIMODAL-SINGLE,TEXT-SINGLE,MULTIMODAL_RERANK

**T2 response:**
```
Query signals: <total planes, American Airlines, seating capacity, December 31, 2017>

Observed retrieval evidence: MR dense recall Top-5 (GT=relevant page, xx=not relevant) and MD dense Top-5 (GT=relevant page, xx=not relevant).

Arm comparison: MR(nDCG=1.000) vs MD(nDCG=1.000). The evidence shows that both MR and MD retrieve the same set of relevant documents with top-ranked being the relevant page about AAL/2017/page_33.pdf. 

Uncertainty: The evidence does not explain the difference in performance between MR and MD since they both achieve the same nDCG@5 score.
```

> Why credible: cites specific nDCG/GT facts, avoids unsupported causal claims.


### C2 | MR_win | MR=1.000 vs TR=0.631

**Query:** What was the largest annual change in unrealized gain/loss on derivative instruments, as reported in the notes to consolidated financial statements fo

**Evidence arms:** MULTIMODAL-SINGLE,TEXT-SINGLE,MULTIMODAL_RERANK

**T2 response:**
```
Query signals: <financial data, specific year (2006), Apple Inc., unrealized gain/loss, derivative instruments, table in notes>

Observed retrieval evidence: MR final rerank nDCG@5=0.0, MR rerank Top-5 (GT=relevant page, xx=not relevant): 1. [xx] JPM/2007/page_164.pdf, 2. [xx] MSI/2008/page_110.pdf

Arm comparison: MR(nDCG=1.000) vs TR(nDCG=0.631). Use only provided ranking evidence; distinguish observed facts from hypotheses.

Uncertainty: The evidence suggests that MR successfully retrieved the relevant page (JPM/2007/page_164.pdf) with a perfect nDCG@5 score, indicating it met the query's needs for financial data and specific conditions. In contrast, TR's performance is lower, possibly due to its reliance on token-level late-interaction reranking which may struggle with the fine-grained multi-condition text required by the query. The observed evidence does not fully explain the difference, but it provides strong support for MR's superior performance.
```

> Why credible: cites specific nDCG/GT facts, avoids unsupported causal claims.


### C3 | nonMR_win | MR=0.000 vs BM=1.000

**Query:** What was the percentage change in Apple Inc.'s inventory from $270 million in 2006 to $346 million in 2007, as reported in the company's 2007 financia

**Evidence arms:** MULTIMODAL-SINGLE,TEXT-SINGLE,MULTIMODAL_RERANK

**T2 response:**
```
Query signals: <financial data, percentage change, Apple Inc., 2006-2007 inventory, financial statements>

Observed retrieval evidence: MR final rerank nDCG@5=0.0, with top-5 results being irrelevant pages (xx).

Arm comparison: MR(nDCG=0.000) vs BM(nDCG=1.000). The MR method failed to find relevant pages, while BM achieved perfect relevance.

Uncertainty: The evidence suggests that MR did not retrieve any relevant pages, which likely explains the low nDCG score. However, it does not definitively rule out the possibility of MR failing due to other factors unrelated to the query specifics.
```

> Why credible: cites specific nDCG/GT facts, avoids unsupported causal claims.


### C4 | tie | MR=1.000 vs TR=1.000

**Query:** Over the six-year period starting from September 30, 2008, and ending on September 30, 2013, by how much did Apple Inc.'s cumulative total shareholder

**Evidence arms:** MULTIMODAL-SINGLE,TEXT-SINGLE,MULTIMODAL_RERANK

**T2 response:**
```
Query signals: <cumulative total shareholder return, Apple Inc., S&P Computer Hardware Index, time period, initial investment>
Observed retrieval evidence: MR dense recall Top-5 (GT=relevant page, xx=not relevant), TR dense recall Top-5 (GT=relevant page, xx=not relevant).
Arm comparison: MR(nDCG=1.000) vs TR(nDCG=1.000). Use only provided ranking evidence; distinguish observed facts from hypotheses.
Uncertainty: The provided ranking evidence does not fully explain the difference in nDCG@5 between MR and TR, as both achieve perfect nDCG@5 scores. However, MR's dense recall suggests it may have better coverage of relevant documents, while TR's token-level late-interaction rerank could be compensating for less dense recall.
```

> Why credible: cites specific nDCG/GT facts, avoids unsupported causal claims.


### C5 | MR_win | MR=1.000 vs TD=0.631

**Query:** What percentage of Apple Inc.'s future minimum lease payments under noncancelable operating leases, as disclosed in the 2014 Form 10-K report, are due

**Evidence arms:** MULTIMODAL-SINGLE,TEXT-SINGLE,MULTIMODAL_RERANK

**T2 response:**
```
Query signals: <Apple Inc., future minimum lease payments, noncancelable operating leases, 2014 Form 10-K report, percentage due after 2019>

Observed retrieval evidence: MR dense recall Top-5 (GT=relevant page, xx=not relevant): 1. [xx] AAPL/2013/page_78.pdf, 2. [xx] AAPL/2012/page_71.pdf, 3. [GT] AAPL/2014/page_78.pdf, 4. [xx] AAPL/2015/page_68.pdf, 5. [xx] UA/2011/page_70.pdf; TD dense Top-5 (GT=relevant page, xx=not relevant): 1. [xx] AON/2007/page_185.pdf, 2. [GT] AAPL/2014/page_78.pdf, 3. [xx] AAPL/2013/page_78.pdf, 4. [xx] AAPL/2007/page_83.pdf, 5. [xx] AAPL/2015/page_68.pdf

Arm comparison: MR(nDCG=1.000) vs TD(nDCG=0.631). MR outperforms TD significantly, indicating that MR's dense multimodal representation and visual late-interaction rerank are crucial for this query.

Uncertainty: The evidence suggests that MR's multimodal dense recall and visual late-interaction rerank effectively capture the relevant information needed to answer the
```

> Why credible: cites specific nDCG/GT facts, avoids unsupported causal claims.


### C6 | nonMR_win | MR=0.631 vs MD=1.000

**Query:** What is the average yearly amortization rate for trademarks, given that they have a weighted average useful life of 7 years as reported by Adobe Inc. 

**Evidence arms:** MULTIMODAL-SINGLE,TEXT-SINGLE,MULTIMODAL_RERANK

**T2 response:**
```
Query signals: <financial data, trademark amortization, Adobe Inc. 2011 financial statements, weighted average useful life of 7 years>

Observed retrieval evidence: MR dense recall Top-5 (GT=relevant page, xx=not relevant):
1. [GT] ADBE/2011/page_83.pdf
2. [xx] ADBE/2013/page_68.pdf
3. [xx] ADBE/2012/page_87.pdf
4. [xx] ADBE/2014/page_70.pdf
5. [xx] ADBE/2009/page_81.pdf

Arm comparison: MR(nDCG=0.631) vs MD(nDCG=1.000). The MR method, despite its multimodal dense recall, performs poorly compared to the MM-Dense method, which retrieves the relevant page directly without needing reranking.

Uncertainty: The evidence does not fully explain the difference, as both methods retrieve the same relevant page (ADBE/2011/page_83.pdf) but MR's poor performance suggests it may struggle with the complex and varied context needed for this query, possibly due to limitations in multimodal interaction or reranking effectiveness.
```

> Why credible: cites specific nDCG/GT facts, avoids unsupported causal claims.


### C7 | nonMR_win | MR=0.631 vs MD=1.000

**Query:** What was the percentage change in the net loss as reported by American Tower from 2001 to 2002, as illustrated in the table that applies the fair valu

**Evidence arms:** MULTIMODAL-SINGLE,TEXT-SINGLE,MULTIMODAL_RERANK

**T2 response:**
```
Query signals: <American Tower, net loss, percentage change, 2001-2002, fair value recognition, SFAS No. 123, Black-Scholes, table>
Observed retrieval evidence: MR ranked [GT] AMT/2002/page_74.pdf at position 1 and [xx] AMT/2004/page_76.pdf at position 2, while MD ranked [GT] AMT/2002/page_74.pdf at position 1. The other pages were not relevant (xx).
Arm comparison: MR(nDCG=0.631) vs MD(nDCG=1.000). The MR method ranked the relevant page higher but still missed the top spot compared to MD.
Uncertainty: The evidence does not fully explain the difference, as MR's lower nDCG could be due to its reliance on multimodal dense representation rather than late interaction reranking, which might have improved precision for this specific query.
```

> Why credible: cites specific nDCG/GT facts, avoids unsupported causal claims.


### C8 | tie | MR=1.000 vs MD=1.000

**Query:** In the 2003 financial records of American Tower, what percentage of the total net proceeds did the $0.1 million loss on disposal represent when the co

**Evidence arms:** MULTIMODAL-SINGLE,TEXT-SINGLE,MULTIMODAL_RERANK

**T2 response:**
```
Query signals: <financial records, American Tower, 2003, office building sale, percentage loss, Schaumburg, Illinois, March 2003>

Observed retrieval evidence: 
- MR final rerank nDCG@5=0.0, indicating it did not improve relevance despite dense multimodal representation.
- MD dense Top-5 (GT=relevant page, xx=not relevant): [GT] AMT/2005/page_83.pdf at rank 1, suggesting strong relevance to the query.

Arm comparison: MR(nDCG=1.000) vs MD(nDCG=1.000). The evidence shows that while both MR and MD achieve perfect nDCG@5, MD's dense multimodal approach successfully retrieves the relevant page (AMT/2005/page_83.pdf) at rank 1, whereas MR's reranking does not improve over its initial dense representation.

Uncertainty: The evidence does not fully explain the difference between MR and MD. While MR's reranking failed to enhance relevance, MD's dense multimodal processing appears to have directly identified the relevant document, suggesting that the multimodal nature of MD is crucial for this specific query involving financial records with detailed computations and visual elements.
```

> Why credible: cites specific nDCG/GT facts, avoids unsupported causal claims.


### C9 | MR_win | MR=1.000 vs TR=0.631

**Query:** If Aon's HR Solutions segment generated approximately 25% of the company's consolidated total revenues in 2010, and the revenue for HR Solutions in 20

**Evidence arms:** MULTIMODAL-SINGLE,TEXT-SINGLE,MULTIMODAL_RERANK

**T2 response:**
```
Query signals: <company Aon, HR Solutions segment, 2010 revenues, total revenues calculation>

Observed retrieval evidence: MR dense recall Top-5 (GT=relevant page, xx=not relevant), with AON/2010/page_55.pdf at rank 3 as the GT document.

Arm comparison: MR(nDCG=1.000) vs TR(nDCG=0.631). The MR method retrieves the most relevant document directly (AON/2010/page_55.pdf) at rank 3, which is crucial for the computation needed to solve the problem. In contrast, TR requires additional reranking steps that do not fully utilize the initial dense recall, leading to a lower nDCG score.

Uncertainty: The evidence suggests that MR successfully identifies the key document needed for the computation, explaining its higher nDCG score compared to TR. However, it does not account for the full potential of other methods like MD and MR-Rerank, which might have performed better if they had access to more refined reranking.
```

> Why credible: cites specific nDCG/GT facts, avoids unsupported causal claims.


### C10 | tie | MR=1.000 vs MD=1.000

**Query:** What was the change in the future annual commitments related to minimum quantities of purchased water between 2019 and 2020 for American Water Works?

**Evidence arms:** MULTIMODAL-SINGLE,TEXT-SINGLE,MULTIMODAL_RERANK

**T2 response:**
```
Query signals: <American Water Works, future annual commitments, minimum quantities of purchased water, change between 2019 and 2020>

Observed retrieval evidence: MR dense recall Top-5 (GT=relevant page, xx=not relevant) and MD dense Top-5 (GT=relevant page, xx=not relevant).

Arm comparison: MR(nDCG=1.000) vs MD(nDCG=1.000). The evidence shows that both MR and MD retrieve the same relevant page (AWK/2018/page_172.pdf) at the top position, indicating they perform equally well in this case.

Uncertainty: The evidence does not fully explain the slight difference in performance, as both methods achieve perfect nDCG@5. However, MR's multimodal dense representation might slightly outperform MD due to its ability to handle more complex modalities, though this is not definitively shown by the provided evidence.
```

> Why credible: cites specific nDCG/GT facts, avoids unsupported causal claims.


---

## 2. Fake (Post-Hoc Rationalization) Explanations

These are T1 outputs (nDCG only). The model invents capability-based causal stories despite having no retrieval evidence beyond the score.


### F1 | tie | MR=1.000 vs MD=1.000

**Query:** What percentage of American Airlines' total planes, as of December 31, 2017, had an average seating capacity of fewer than 100 passengers?

**T1 response:**
```
Query signals: <computation/conditions, entities, visual data>
Observed retrieval evidence: nDCG scores only; no page/ranking details.
Arm comparison: MR(nDCG=1.000) vs MD(nDCG=1.000). The gap is fact; it could be due to hypotheses such as differences in how they handle complex computations and conditions.
Uncertainty: The nDCG scores indicate that both MR and MD perform equally well, but the exact reasons for their performance cannot be fully explained without additional details about how they process the query and its conditions.
```

> Why fake: causal claims ('likely... through multimodal interaction') not grounded in any provided evidence.


### F2 | MR_win | MR=1.000 vs TR=0.631

**Query:** What was the largest annual change in unrealized gain/loss on derivative instruments, as reported in the notes to consolidated financial statements fo

**T1 response:**
```
Query signals: <financial data, specific year (2006), Apple Inc., unrealized gain/loss, derivative instruments, financial statement notes, table summarization of other comprehensive income>
Observed retrieval evidence: nDCG scores only; no page/ranking details.
Arm comparison: MR(nDCG=1.000) vs TR(nDCG=0.631). The gap is fact; it suggests MR can handle multimodal dense representation and visual interaction better than TR which relies more on text-dense recall and token-level late interaction.
Uncertainty: The gap could be due to MR's ability to effectively process multimodal data, including tables and potentially visual elements within the financial report, while TR struggles with the detailed, structured information required for such a complex query. Additionally, the high nDCG score for MR might indicate that the top-ranked documents are highly relevant and well-positioned, whereas TR's lower score could reflect issues with ranking precision or relevance
```

> Why fake: causal claims ('likely... through multimodal interaction') not grounded in any provided evidence.


### F3 | nonMR_win | MR=0.000 vs BM=1.000

**Query:** What was the percentage change in Apple Inc.'s inventory from $270 million in 2006 to $346 million in 2007, as reported in the company's 2007 financia

**T1 response:**
```
Query signals: <financial data, percentage change, inventory, 2006-2007, numerical values>
Observed retrieval evidence: nDCG scores only; no page/ranking details.
Arm comparison: MR(nDCG=0.000) vs BM(nDCG=1.000). The gap is fact; it could be due to MR's multimodal focus missing the exact numerical values and BM's strong keyword matching excelling here.
```

> Why fake: causal claims ('likely... through multimodal interaction') not grounded in any provided evidence.


### F4 | tie | MR=1.000 vs TR=1.000

**Query:** Over the six-year period starting from September 30, 2008, and ending on September 30, 2013, by how much did Apple Inc.'s cumulative total shareholder

**T1 response:**
```
Query signals: <financial data, time period, company performance, index comparison, numerical calculation>
Observed retrieval evidence: nDCG scores only; no page/ranking details.
Arm comparison: MR(nDCG=1.000) vs TR(nDCG=1.000). The gap is fact; it could be due to hypotheses such as:
- MR's multimodal dense representation might better capture complex financial charts and detailed historical data.
- TR's token-level late interaction might refine the search to more closely match the specific numerical and temporal criteria in the query.
```

> Why fake: causal claims ('likely... through multimodal interaction') not grounded in any provided evidence.


### F5 | MR_win | MR=1.000 vs TD=0.631

**Query:** What percentage of Apple Inc.'s future minimum lease payments under noncancelable operating leases, as disclosed in the 2014 Form 10-K report, are due

**T1 response:**
```
Query signals: <Apple Inc., future minimum lease payments, noncancelable operating leases, 2014 Form 10-K, percentage, year 2019>

Observed retrieval evidence: nDCG scores only; no page/ranking details.

Arm comparison: MR(nDCG=1.000) vs TD(nDCG=0.631). The gap is fact; the hypothesis is that MR's strong multimodal capabilities allow it to better understand and retrieve the specific financial data needed, while TD struggles with the precise numerical and temporal information required.

Uncertainty: The gap can be explained by the hypothesis that MR's multimodal approach allows it to effectively handle the complex, detailed nature of the query, including the need to parse and understand financial data from a document, which TD may not fully capture due to its focus on dense text representation without additional modalities. Additionally, the high nDCG score of MR suggests it ranks relevant documents higher
```

> Why fake: causal claims ('likely... through multimodal interaction') not grounded in any provided evidence.


### F6 | nonMR_win | MR=0.631 vs MD=1.000

**Query:** What is the average yearly amortization rate for trademarks, given that they have a weighted average useful life of 7 years as reported by Adobe Inc. 

**T1 response:**
```
Query signals: <financial data, trademark amortization, weighted average useful life, Adobe Inc., 2011 financial statements>
Observed retrieval evidence: nDCG scores only; no page/ranking details.
Arm comparison: MR(nDCG=0.631) vs MD(nDCG=1.000). The gap is fact; it could be due to MR's reliance on multimodal dense representation potentially missing key financial nuances, while MD's dense multimodal approach might better capture the required financial context.
```

> Why fake: causal claims ('likely... through multimodal interaction') not grounded in any provided evidence.


### F7 | nonMR_win | MR=0.631 vs MD=1.000

**Query:** What was the percentage change in the net loss as reported by American Tower from 2001 to 2002, as illustrated in the table that applies the fair valu

**T1 response:**
```
Query signals: <entities: American Tower, net loss, percentage change, 2001 to 2002; conditions: fair value recognition provisions of SFAS No. 123, Black-Scholes option-pricing model; needs: computation and visual information>

Observed retrieval evidence: nDCG scores only; no page/ranking details.

Arm comparison: MR(nDCG=0.631) vs MD(nDCG=1.000). The gap is fact; the hypothesis is that MR's lower nDCG score could be due to its reliance on multimodal dense representations, which might struggle with the specific textual and visual requirements of the query, while MD's dense multimodal representation aligns better with the query's needs.

Uncertainty: The gap between MR and MD can be explained by the specific demands of the query. MR's multimodal approach may not fully capture the nuanced textual and visual elements required to accurately rank documents
```

> Why fake: causal claims ('likely... through multimodal interaction') not grounded in any provided evidence.


### F8 | tie | MR=1.000 vs MD=1.000

**Query:** In the 2003 financial records of American Tower, what percentage of the total net proceeds did the $0.1 million loss on disposal represent when the co

**T1 response:**
```
Query signals: <financial records, 2003, American Tower, net proceeds, $0.1 million loss, office building, Schaumburg, Illinois, March 2003>
Observed retrieval evidence: nDCG scores only; no page/ranking details.
Arm comparison: MR(nDCG=1.000) vs MD(nDCG=1.000). The gap is fact; it is explained as hypotheses.
Hypotheses: The strong performance of both MR and MD could be due to their ability to handle multimodal data effectively, which might include financial documents with visual elements like charts or tables. Additionally, they excel in complex queries that require understanding of context and detailed information, such as the specific transaction details and location mentioned. However, without detailed page/ranking analysis, the exact reasons for their superior performance cannot be fully determined.
```

> Why fake: causal claims ('likely... through multimodal interaction') not grounded in any provided evidence.


### F9 | MR_win | MR=1.000 vs TR=0.631

**Query:** If Aon's HR Solutions segment generated approximately 25% of the company's consolidated total revenues in 2010, and the revenue for HR Solutions in 20

**T1 response:**
```
Query signals: <financial data, percentage calculation, revenue figures, year 2010>
Observed retrieval evidence: nDCG scores only; no page/ranking details.
Arm comparison: MR(nDCG=1.000) vs TR(nDCG=0.631). The gap is fact; hypotheses include:
- MR likely excels due to its multimodal dense representation, which could better capture the financial context and numerical precision needed for the computation.
- TR might struggle with the precise numerical calculation required, despite its strong performance in other contexts like semantic similarity.
```

> Why fake: causal claims ('likely... through multimodal interaction') not grounded in any provided evidence.


### F10 | tie | MR=1.000 vs MD=1.000

**Query:** What was the change in the future annual commitments related to minimum quantities of purchased water between 2019 and 2020 for American Water Works?

**T1 response:**
```
Query signals: <conditions, computation>
Observed retrieval evidence: nDCG scores only; no page/ranking details.
Arm comparison: MR(nDCG=1.000) vs MD(nDCG=1.000). The gap is fact; it could be due to hypotheses such as:
- Differences in how well the models handle the specific numerical data and temporal changes required by the query.
- Variations in how effectively the models capture the context around the key terms "minimum quantities" and "purchased water."
- Potential differences in how well the models can integrate the visual elements (if any) that might be present in the documents.
Uncertainty: The exact reasons for the slight difference in nDCG scores cannot be fully explained solely based on the provided information. It may require further analysis of the actual documents retrieved by each model to identify specific discrepancies.
```

> Why fake: causal claims ('likely... through multimodal interaction') not grounded in any provided evidence.


---

## 3. Counter-Control: Similar Queries, Opposite Winners

T1 uses the same capability script regardless of who won, confirming post-hoc rationalization.


---
## 3. Counter-Control: Similar Queries, Opposite Winners
T1 uses the same capability script regardless of who won, confirming post-hoc rationalization.

### Pair 1 (keyword overlap=76%, dataset=finqa)

- **MR-win:** What percentage of the total minimum lease payments for Union Pacific Corporation as of De
  - MR=1.000, MULTIMOD=0.631
- **nonMR-win:** What percentage of the total minimum lease payments, which amount to $5,652 million as of 
  - MR=0.000, MULTIMOD=0.631

**T1 MR-win Arm comparison:** Arm comparison: MR(nDCG=1.000) vs MD(nDCG=0.631). The gap is fact; it could be due to MR's strong multimodal capabilities and dense representation, potentially capturing the comple

**T1 nonMR-win Arm comparison:** Arm comparison: MR(nDCG=0.000) vs MD(nDCG=0.631). The gap is fact; it could be due to hypotheses such as:
- MR's multimodal approach might struggle with the precise numerical and f

> Both invoke the same 'multimodal dense + visual interaction' script, reversed to fit the winner. Post-hoc rationalization confirmed.

### Pair 2 (keyword overlap=64%, dataset=finqa)

- **MR-win:** What percentage of the total minimum lease payments for Union Pacific Corporation as of De
  - MR=1.000, MULTIMOD=0.631
- **nonMR-win:** What percentage of the total minimum operating lease payments due from Union Pacific Corpo
  - MR=0.000, TEXT-SIN=0.387

**T1 MR-win Arm comparison:** Arm comparison: MR(nDCG=1.000) vs MD(nDCG=0.631). The gap is fact; it could be due to MR's strong multimodal capabilities and dense representation, potentially capturing the comple

**T1 nonMR-win Arm comparison:** Arm comparison: MR(nDCG=0.000) vs TD(nDCG=0.387). The gap is fact; it could be due to hypotheses such as:
- TD's strong semantic similarity might better capture the financial conte

> Both invoke the same 'multimodal dense + visual interaction' script, reversed to fit the winner. Post-hoc rationalization confirmed.

### Pair 3 (keyword overlap=56%, dataset=vqaonbd)

- **MR-win:** What is the minimum value between operating income and interest income, net for Monster Be
  - MR=0.387, MULTIMOD=0.000
- **nonMR-win:** What is the minimum value between charge-offs and recoveries for Discover Financial in 200
  - MR=0.631, BM25=1.000

**T1 MR-win Arm comparison:** Arm comparison: MR(nDCG=0.387) vs MD(nDCG=0.000). The gap is fact; the hypothesis is that MR's multimodal approach better handles the complex query involving both financial data an

**T1 nonMR-win Arm comparison:** Arm comparison: MR(nDCG=0.631) vs BM(nDCG=1.000). The gap is fact; it could be due to BM's strong keyword matching ability, MR's multimodal dense representation, or MR's visual lat

> Both invoke the same 'multimodal dense + visual interaction' script, reversed to fit the winner. Post-hoc rationalization confirmed.

