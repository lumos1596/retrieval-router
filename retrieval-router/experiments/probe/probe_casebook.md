# 探针实验坏例对照

## Local 修复了 Query-only 错误的案例（10条）

**1. [tatdqa]** What was the cash provided by financing activities for hc2-holdings-inc in 2019?
- Router→MD(nDCG=0.63)  Local→MR(nDCG=1.00)  Oracle=MR
- 五臂nDCG: {'MR': 1.0, 'MD': 0.6309297680854797, 'TR': 0.0, 'TD': 0.0, 'BM': 0.0}
- 最近探针: ['What was the amount of interest attracted by cash balances i', "What is the fair value of the company's total assets as at D", "What is the percentage change in the company's asset-backed "]

**2. [tatdqa]** What is the difference between the largest and smallest Value Realized on Vesting of RSUs and RSAs?
- Router→TR(nDCG=1.00)  Local→MR(nDCG=1.00)  Oracle=MR
- 五臂nDCG: {'MR': 1.0, 'MD': 0.5, 'TR': 1.0, 'TD': 0.0, 'BM': 1.0}
- 最近探针: ['What was the change in the Performance-Based RSUs between 20', 'How is the basic earnings per share calculated?', 'What was the percentage change in balance of restricted stoc']

**3. [tatdqa]** What percentage of the total property and equipment at cost does 'Equipment and software' represent for parkervision in 2019, before accounting for depreciation?
- Router→MD(nDCG=0.00)  Local→MR(nDCG=1.00)  Oracle=MR
- 五臂nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 0.0, 'TD': 0.0, 'BM': 0.0}
- 最近探针: ['What was the depreciation expense on property and equipment ', 'What is the change in Depreciation and amortization between ', 'What was the percentage change in interest expense from 2018']

**4. [tatdqa]** What is the respective revenue from Europe in 2018 and 2019?
- Router→MD(nDCG=0.00)  Local→MR(nDCG=1.00)  Oracle=MR
- 五臂nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 0.0, 'TD': 0.0, 'BM': 0.0}
- 最近探针: ['What is the average Total cost of revenue for 2017-2018?', 'What is the total operating revenue recorded in 2019?', 'What was the change in the amount within one year between 20']

**5. [tatdqa]** What was the change in Amortization of actuarial gains and losses between 2017 and 2018?
- Router→BM(nDCG=0.00)  Local→MR(nDCG=0.63)  Oracle=MR
- 五臂nDCG: {'MR': 0.6309297680854797, 'MD': 0.0, 'TR': 0.0, 'TD': 0.0, 'BM': 0.0}
- 最近探针: ['What is the change in Depreciation and amortization between ', 'What was the percentage change in interest expense from 2018', 'What was the increase / (decrease) in the current assets fro']

**6. [tatdqa]** What is the net income as at June 30, 2019?
- Router→MR(nDCG=0.00)  Local→MD(nDCG=0.63)  Oracle=MD
- 五臂nDCG: {'MR': 0.0, 'MD': 0.6309297680854797, 'TR': 0.0, 'TD': 0.0, 'BM': 0.0}
- 最近探针: ['How much net income represented diluted earnings per share i', 'As of June 30, 2018, what is the value of the certificates o', 'What is the percentage change for the weighted-average ordin']

**7. [tatdqa]** What is the increase in cloud subscription revenue in America in 2019?
- Router→MD(nDCG=0.43)  Local→MR(nDCG=1.00)  Oracle=MR
- 五臂nDCG: {'MR': 1.0, 'MD': 0.43067654967308044, 'TR': 1.0, 'TD': 0.43067654967308044, 'BM': 0.0}
- 最近探针: ['What is the average percentage of revenue earned in North Am', 'What was the percentage change in cost of Online Advertising', 'What is the average Total cost of revenue for 2017-2018?']

**8. [tatdqa]** What would be the number of outstanding options whose exercise price are between $8.04 to $15.06?
- Router→BM(nDCG=1.00)  Local→MR(nDCG=1.00)  Oracle=MR
- 五臂nDCG: {'MR': 1.0, 'MD': 1.0, 'TR': 1.0, 'TD': 0.0, 'BM': 1.0}
- 最近探针: ['What was the number of options granted in 2019?', 'What is the change in the Exercised during the period weight', 'What are the activity types involving stock option awards?']

**9. [tatdqa]** What was the ratio of Federal NOL to Foreign NOL for the period 2030 to 2034, given the Federal NOL was $3.6 and there was no Foreign NOL?
- Router→MR(nDCG=0.50)  Local→MD(nDCG=0.63)  Oracle=MD
- 五臂nDCG: {'MR': 0.5, 'MD': 0.6309297680854797, 'TR': 0.0, 'TD': 0.43067654967308044, 'BM': 0.0}
- 最近探针: ['What is the Post-tax loss of  discontinued operations in 201', 'What was the amount of cumulative acquisition-related expens', 'What was the change in Foreign currency losses recorded in A']

**10. [tatdqa]** What was the net cash inflow from operating activities for BT Group plc in the year ended 31 March 2019?
- Router→TD(nDCG=1.00)  Local→MR(nDCG=1.00)  Oracle=MR
- 五臂nDCG: {'MR': 1.0, 'MD': 0.6309297680854797, 'TR': 1.0, 'TD': 1.0, 'BM': 0.0}
- 最近探针: ['What is the net book value as of 31 March in 2019?', 'What was the amount of interest attracted by cash balances i', 'What is the total operating revenue recorded in 2019?']

## Local 新造成错误的案例（10条）

**1. [tatdqa]** What is the percentage change in plan asset fair value allocated in Canada equity securities between 2018 and 2019?
- Router→MD(nDCG=1.00)  Local→MR(nDCG=0.43)  Oracle=MD
- 五臂nDCG: {'MR': 0.43067654967308044, 'MD': 1.0, 'TR': 0.0, 'TD': 0.43067654967308044, 'BM': 1.0}
- 最近探针: ["What is the percentage change in the company's asset-backed ", 'What was the percentage change in balance of restricted stoc', 'What was the percentage change in interest expense from 2018']

**2. [tatdqa]** What is the percentage change in the gain on bargain purchase from 2018 to 2019?
- Router→MD(nDCG=1.00)  Local→MR(nDCG=0.63)  Oracle=MD
- 五臂nDCG: {'MR': 0.6309297680854797, 'MD': 1.0, 'TR': 0.6309297680854797, 'TD': 0.0, 'BM': 0.43067654967308044}
- 最近探针: ['What was the percentage change in interest expense from 2018', 'What was the change in the amount within one year between 20', 'What was the percentage change in balance of restricted stoc']

**3. [tatdqa]** What was the decline in the global financing receivables?
- Router→MR(nDCG=0.63)  Local→MD(nDCG=0.39)  Oracle=MR
- 五臂nDCG: {'MR': 0.6309297680854797, 'MD': 0.38685280084609985, 'TR': 0.0, 'TD': 0.0, 'BM': 0.43067654967308044}
- 最近探针: ['What is the Post-tax loss of  discontinued operations in 201', 'What was the total amount of trade receivables and contract ', 'What was the change in Foreign currency losses recorded in A']

**4. [tatdqa]** What is the average value of Building and leasehold improvements for the years 2018 and 2019 in the property, plant and equipment of hc2-holdings-inc?
- Router→MD(nDCG=1.00)  Local→MR(nDCG=0.63)  Oracle=MD
- 五臂nDCG: {'MR': 0.6309297680854797, 'MD': 1.0, 'TR': 0.6309297680854797, 'TD': 0.0, 'BM': 0.0}
- 最近探针: ['What were the lease costs in 2019?', 'What was the change in the land value from 2018 to 2019?', 'What was the increase in construction in progress for Centur']

**5. [tatdqa]** How much does operating leases account for total contractual cash obligations for period of less than 1 year?
- Router→MD(nDCG=0.63)  Local→MR(nDCG=0.00)  Oracle=MD
- 五臂nDCG: {'MR': 0.0, 'MD': 0.6309297680854797, 'TR': 0.5, 'TD': 0.0, 'BM': 0.0}
- 最近探针: ['How many years did Interest expense on capital leases exceed', 'How many lease liabilities for operating leases exceeded $50', 'Which years does the table provide information for the compa']

**6. [tatdqa]** Which key performance indicator accounts for two thirds in Growth?
- Router→MR(nDCG=1.00)  Local→MD(nDCG=1.00)  Oracle=MR
- 五臂nDCG: {'MR': 1.0, 'MD': 1.0, 'TR': 1.0, 'TD': 0.0, 'BM': 0.6309297680854797}
- 最近探针: ['What proportion of level 2 inputs is made up of money market', 'What was the increase / (decrease) in the current assets fro', 'What was the change in the Performance-Based RSUs between 20']

**7. [tatdqa]** What is the net sales of software in 2019 and 2018 respectively?
- Router→MD(nDCG=0.62)  Local→MR(nDCG=0.00)  Oracle=MD
- 五臂nDCG: {'MR': 0.0, 'MD': 0.624050498008728, 'TR': 0.0, 'TD': 0.5706416964530945, 'BM': 0.0}
- 最近探针: ['What was the change in the Capitalized software development ', 'What is the change in share of annual sales between 2018 and', 'What is the average Total cost of revenue for 2017-2018?']

**8. [tatdqa]** What was the Restricted cash in 2017?
- Router→MR(nDCG=0.00)  Local→MD(nDCG=0.00)  Oracle=MR
- 五臂nDCG: {'MR': 0.0, 'MD': 0.0, 'TR': 0.0, 'TD': 0.0, 'BM': 0.0}
- 最近探针: ['In which year  was Restricted cash included in other assets,', 'What was the amount of interest attracted by cash balances i', 'What was the percentage change in balance of restricted stoc']

**9. [tatdqa]** What was the difference between Net income from continuing operations and Income before income taxes?
- Router→MR(nDCG=0.00)  Local→MD(nDCG=0.00)  Oracle=MR
- 五臂nDCG: {'MR': 0.0, 'MD': 0.0, 'TR': 0.0, 'TD': 0.0, 'BM': 0.0}
- 最近探针: ['What was the change in Other deferred tax liabilities for Ne', 'How much net income represented diluted earnings per share i', 'What is the Post-tax loss of  discontinued operations in 201']

**10. [tatdqa]** What is the percentage decrease in Micron Technology Inc.'s total stock-based compensation expense from 2017 to 2018, as reflected in the stock-based compensation expense table for the years ended 2017, 2018, and 2019?
- Router→MR(nDCG=1.00)  Local→MD(nDCG=0.00)  Oracle=MR
- 五臂nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 0.0, 'BM': 0.0}
- 最近探针: ['What is the change in research and development stock-based c', "What is the percentage change in the company's asset-backed ", 'What was the percentage change in interest expense from 2018']
