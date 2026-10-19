# Training objective badcases (Wiki-SS)

## B1 improved vs B0 (top 10 by regret drop)

**1.** in the 1st century bc egypt became a colony of
- B0=MD(r=1.000)  B1=MR(r=0.000)  B2=MR(r=0.000)
- nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 0.431, 'BM': 1.0}

**2.** who does luke kennard play for in the nba
- B0=MD(r=1.000)  B1=MR(r=0.000)  B2=MR(r=0.000)
- nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 0.387, 'BM': 0.387}

**3.** which version of windows introduced the metro user interface
- B0=MR(r=1.000)  B1=MD(r=0.000)  B2=MD(r=0.000)
- nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 1.0, 'TD': 1.0, 'BM': 1.0}

**4.** when was take me out to the ballgame first sung
- B0=MD(r=1.000)  B1=MR(r=0.000)  B2=MR(r=0.000)
- nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 0.0, 'BM': 0.0}

**5.** who won the 2017 french open men 's singles
- B0=TR(r=1.000)  B1=MD(r=0.000)  B2=MR(r=0.613)
- nDCG: {'MR': 0.387, 'MD': 1.0, 'TR': 0.0, 'TD': 1.0, 'BM': 0.431}

**6.** when did dude where 's my car come out
- B0=MD(r=1.000)  B1=BM(r=0.000)  B2=TR(r=0.000)
- nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 0.0, 'BM': 1.0}

**7.** when did hot wheels start super treasure hunts
- B0=BM(r=1.000)  B1=MR(r=0.000)  B2=BM(r=1.000)
- nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 0.431, 'TD': 0.0, 'BM': 0.0}

**8.** where is the home and family show filmed
- B0=MD(r=1.000)  B1=MR(r=0.000)  B2=MR(r=0.000)
- nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 1.0, 'BM': 0.0}

**9.** what time is game of thrones based on
- B0=MD(r=1.000)  B1=MR(r=0.000)  B2=MR(r=0.000)
- nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 0.0, 'BM': 0.387}

**10.** who wins big brother battle back season 19
- B0=MD(r=1.000)  B1=MR(r=0.000)  B2=MR(r=0.000)
- nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 0.431, 'TD': 0.631, 'BM': 0.431}

## B1 worsened vs B0 (top 10 by regret increase)

**1.** who does betty draper have an affair with
- B0=MD(r=0.000)  B1=BM(r=1.000)  B2=MD(r=0.000)
- nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 0.0, 'TD': 0.0, 'BM': 0.0}

**2.** who was golden states coach before steve kerr
- B0=MD(r=0.000)  B1=BM(r=1.000)  B2=MD(r=0.000)
- nDCG: {'MR': 1.0, 'MD': 1.0, 'TR': 1.0, 'TD': 0.0, 'BM': 0.0}

**3.** who owns the lord of the rings franchise
- B0=MD(r=0.000)  B1=MR(r=1.000)  B2=MR(r=1.000)
- nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 0.0, 'TD': 1.0, 'BM': 1.0}

**4.** who does christine choose in love never dies
- B0=MD(r=0.000)  B1=MR(r=1.000)  B2=MR(r=1.000)
- nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 0.431, 'TD': 1.0, 'BM': 0.431}

**5.** where was the peace treaty with japan signed
- B0=MD(r=0.000)  B1=TR(r=1.000)  B2=MR(r=1.000)
- nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 0.0, 'TD': 1.0, 'BM': 1.0}

**6.** who were the two black students enrolled at the university
- B0=MD(r=0.000)  B1=MR(r=1.000)  B2=MR(r=1.000)
- nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 0.0, 'TD': 1.0, 'BM': 1.0}

**7.** who plays killer croc in the movie suicide squad
- B0=TR(r=0.000)  B1=MR(r=1.000)  B2=BM(r=0.500)
- nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 1.0, 'TD': 0.431, 'BM': 0.5}

**8.** where is the main office of the ncaa
- B0=TR(r=0.000)  B1=MD(r=1.000)  B2=MR(r=0.000)
- nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 1.0, 'BM': 0.0}

**9.** what is the meaning of ardaas in punjabi
- B0=MD(r=0.000)  B1=MR(r=1.000)  B2=MR(r=1.000)
- nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 1.0, 'TD': 1.0, 'BM': 1.0}

**10.** what is the name of the heroine in bharat ane nenu
- B0=TR(r=0.000)  B1=MD(r=1.000)  B2=TD(r=1.000)
- nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 0.0, 'BM': 1.0}
