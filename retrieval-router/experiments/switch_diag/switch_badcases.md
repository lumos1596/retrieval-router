# Switch badcases (Wiki-SS, model=B2)

## Correct switches (pred!=MR and r_pred > r_MR) — 134 total

**1.** who has won the most basketball ncaa championships
- pred=MD r_pred=1.000 r_MR=0.000 gain=+1.000
- arms: MR=0.000 MD=1.000 TR=0.000 TD=0.500 BM=0.000

**2.** who plays james in james and the giant peach
- pred=TD r_pred=1.000 r_MR=0.000 gain=+1.000
- arms: MR=0.000 MD=1.000 TR=0.631 TD=1.000 BM=1.000

**3.** when did reese 's peanut butter cups start
- pred=MD r_pred=1.000 r_MR=0.500 gain=+0.500
- arms: MR=0.500 MD=1.000 TR=1.000 TD=0.631 BM=1.000

**4.** what is the full name of bb t bank
- pred=TR r_pred=1.000 r_MR=0.631 gain=+0.369
- arms: MR=0.631 MD=1.000 TR=1.000 TD=1.000 BM=1.000

**5.** who sang the song wreck of the edmund fitzgerald
- pred=TD r_pred=0.631 r_MR=0.000 gain=+0.631
- arms: MR=0.000 MD=1.000 TR=0.000 TD=0.631 BM=0.631

**6.** who plays killer croc in the movie suicide squad
- pred=BM r_pred=0.500 r_MR=0.000 gain=+0.500
- arms: MR=0.000 MD=1.000 TR=1.000 TD=0.431 BM=0.500

**7.** who starred in the movie my cousin vinny
- pred=BM r_pred=1.000 r_MR=0.000 gain=+1.000
- arms: MR=0.000 MD=1.000 TR=0.000 TD=1.000 BM=1.000

**8.** when did the dissolution of the monasteries happen
- pred=TR r_pred=0.631 r_MR=0.500 gain=+0.131
- arms: MR=0.500 MD=1.000 TR=0.631 TD=1.000 BM=1.000

**9.** who played bruce banner in the original incredible hulk
- pred=BM r_pred=1.000 r_MR=0.000 gain=+1.000
- arms: MR=0.000 MD=1.000 TR=0.000 TD=1.000 BM=1.000

**10.** who plays gwen on as the world turns
- pred=BM r_pred=1.000 r_MR=0.631 gain=+0.369
- arms: MR=0.631 MD=1.000 TR=1.000 TD=1.000 BM=1.000

## Missed high-value switches (delta>=0.2 but pred=MR) — 335 total

**1.** where do the light reactions of photosynthesis take place
- pred=MR r_MR=0.000 | best_alt=TD r_alt=1.000 delta=1.000
- arms: MR=0.000 MD=0.631 TR=0.000 TD=1.000 BM=0.387

**2.** who is the first president of all india trade union congress
- pred=MR r_MR=0.631 | best_alt=MD r_alt=1.000 delta=0.369
- arms: MR=0.631 MD=1.000 TR=1.000 TD=1.000 BM=1.000

**3.** who sold the most albums in first week
- pred=MR r_MR=0.000 | best_alt=TD r_alt=1.000 delta=1.000
- arms: MR=0.000 MD=0.387 TR=0.000 TD=1.000 BM=0.631

**4.** where was the film far from the madding crowd filmed
- pred=MR r_MR=0.631 | best_alt=MD r_alt=1.000 delta=0.369
- arms: MR=0.631 MD=1.000 TR=1.000 TD=1.000 BM=1.000

**5.** where did the queen mother live after the king died
- pred=MR r_MR=0.631 | best_alt=TR r_alt=1.000 delta=0.369
- arms: MR=0.631 MD=0.631 TR=1.000 TD=0.000 BM=0.500

**6.** who has won the icc world cup of 2011
- pred=MR r_MR=0.000 | best_alt=BM r_alt=0.631 delta=0.631
- arms: MR=0.000 MD=0.000 TR=0.500 TD=0.000 BM=0.631

**7.** who owns the lord of the rings franchise
- pred=MR r_MR=0.000 | best_alt=MD r_alt=1.000 delta=1.000
- arms: MR=0.000 MD=1.000 TR=0.000 TD=1.000 BM=1.000

**8.** which is the only large river of indian desert of rajasthan
- pred=MR r_MR=0.631 | best_alt=MD r_alt=1.000 delta=0.369
- arms: MR=0.631 MD=1.000 TR=0.631 TD=1.000 BM=1.000

**9.** when was the first john wick movie made
- pred=MR r_MR=0.631 | best_alt=MD r_alt=1.000 delta=0.369
- arms: MR=0.631 MD=1.000 TR=0.631 TD=0.000 BM=1.000

**10.** who sings i can go the distance in hercules
- pred=MR r_MR=0.000 | best_alt=TR r_alt=1.000 delta=1.000
- arms: MR=0.000 MD=0.631 TR=1.000 TD=1.000 BM=0.631

## Wrong switches (pred!=MR but r_pred < r_MR) — 131 total

**1.** who played don corleone 's daughter in the godfather
- pred=BM r_pred=0.387 r_MR=1.000 loss=-0.613
- arms: MR=1.000 MD=1.000 TR=1.000 TD=1.000 BM=0.387

**2.** who plays zoey on how i met your mother
- pred=BM r_pred=0.000 r_MR=1.000 loss=-1.000
- arms: MR=1.000 MD=0.000 TR=1.000 TD=1.000 BM=0.000

**3.** when did hot wheels start super treasure hunts
- pred=BM r_pred=0.000 r_MR=1.000 loss=-1.000
- arms: MR=1.000 MD=0.000 TR=0.431 TD=0.000 BM=0.000

**4.** when is the next handmaid 's tale released
- pred=BM r_pred=0.387 r_MR=1.000 loss=-0.613
- arms: MR=1.000 MD=0.000 TR=1.000 TD=0.000 BM=0.387

**5.** who voiced baloo in last years remake of jungle book
- pred=BM r_pred=0.000 r_MR=1.000 loss=-1.000
- arms: MR=1.000 MD=0.000 TR=0.631 TD=0.000 BM=0.000

**6.** who played jamie scott in one tree hill
- pred=BM r_pred=0.631 r_MR=1.000 loss=-0.369
- arms: MR=1.000 MD=0.631 TR=1.000 TD=1.000 BM=0.631

**7.** who killed the governor in the walking dead
- pred=BM r_pred=0.000 r_MR=1.000 loss=-1.000
- arms: MR=1.000 MD=1.000 TR=1.000 TD=0.000 BM=0.000

**8.** whats the name of the place in black panther
- pred=MD r_pred=0.000 r_MR=1.000 loss=-1.000
- arms: MR=1.000 MD=0.000 TR=1.000 TD=0.000 BM=0.000

**9.** who played morticia addams in addams family values
- pred=TD r_pred=0.631 r_MR=1.000 loss=-0.369
- arms: MR=1.000 MD=1.000 TR=1.000 TD=0.631 BM=1.000

**10.** wolf of wall street what did he do illegal
- pred=MD r_pred=0.000 r_MR=0.631 loss=-0.631
- arms: MR=0.631 MD=0.000 TR=1.000 TD=0.000 BM=0.000
