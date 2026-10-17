# distilbert: most confident errors (pooled test folds)

## False positives (label 0, predicted 1)

| prob | source | label origin | text |
|---|---|---|---|
| 0.999 | persona_turn | pseudo | I'm focusing on triangles and their properties. By the way, I'm also coaching the soccer team this season. Do you have any suggestions for incorporating math into our practice sessions? I tend to stick with traditional d |
| 0.999 | persona_turn | pseudo | I've dabbled in social media, but I'm not very tech-savvy. I usually prefer hands-on approaches. Speaking of which, I'm thinking about offering bread-making classes. Do you think that would be a good idea? |
| 0.998 | persona_turn | pseudo | I usually prefer a hot shower, but I'm always open to new ideas. Speaking of which, do you know any good podcasts about healthcare or nursing? I like to stay updated on the latest trends in my field. |
| 0.996 | persona_turn | pseudo | Those are all great ideas! I especially like the idea of mixing up practice routines. I usually prefer to stick to a set schedule, but I can see how some variety could help keep things fresh. Do you have any suggestions  |
| 0.996 | persona_turn | pseudo | Most of my clients prefer move-in ready homes, but some are open to minor renovations if it means getting into a better neighborhood. Speaking of which, what are some key neighborhood features I should be pointing out to |
| 0.995 | persona_turn | pseudo | Thanks for the advice! I'll give those a try. Speaking of breaks, I usually prefer working from home, but I'm thinking about trying out a local coffee shop for a change of scenery. Any thoughts on staying productive in a |
| 0.995 | persona_turn | pseudo | I've dabbled in online marketing, but I'm not great with technology. I prefer hands-on work. Speaking of which, I've been getting more calls about water heater installations lately. Any advice on how to streamline that p |
| 0.994 | persona_turn | pseudo | To be honest, I'm a bit apprehensive about the new system. I prefer the traditional methods we've been using. But I understand the need for progress. On a different note, we're organizing a fundraiser for the pediatric o |
| 0.992 | persona_turn | pseudo | I haven't looked into it much yet. Speaking of creativity, I'm struggling with writer's block. As someone who spends most of their time in a café, I tend to people-watch for inspiration. Any other suggestions for overcom |
| 0.990 | persona_turn | pseudo | That's a great idea! I've been hesitant about going digital, but I can see how it could save us time. Speaking of time, do you have any tips for balancing work and personal life as a small business owner? I usually prefe |
| 0.988 | persona_turn | pseudo | Well, one thing I've learned is to always have backup plans for when things go wrong. You never know when you'll get stuck somewhere due to weather or an accident up ahead. So I keep extra water, non-perishable foods, an |
| 0.987 | oasst | pseudo | I like the red wine vinegar idea, however I only have apple cider vinegar. Will this be a good replacement? |
| 0.984 | oasst | pseudo | It's good but my sister doesn't like coffee, make another version of this recipe that uses some of the ingredients aformentionned, I will make a different one for her and I do not want to spend too much on this. |
| 0.984 | persona_turn | pseudo | I'm really intrigued by UI/UX design. Do you know of any good online courses or resources for learning more about it? I'd prefer video tutorials or interactive lessons over text-based content. |
| 0.984 | question | gold | Winter is coming up and I need recommendations for some stylish yet warm and cozy outerwear options to get me through the cold months. |

## False negatives (label 1, predicted 0)

| prob | source | label origin | text |
|---|---|---|---|
| 0.002 | oasst | audited | which would you say is the healthier option ? |
| 0.002 | oasst | audited | I'm looking for more contemporary sounding names, what about some names that are not of Biblical origin? |
| 0.002 | oasst | audited | Write a poem that says how Middlesbrough FC are better than Sunderland FC. Use colloquial language from the North East of England, such as the term 'Howay'. Include reference to football hooliganism, and how Sunderland f |
| 0.002 | oasst | pseudo | Cool! Can you write a function that does tetration now? Since floats can overflow, I'd like it to convert input into Python's fraction class. |
| 0.002 | persona_turn | audited | I always emphasize the importance of scientifically proven methods first. However, I'm curious about new trends too. Recently, I've been researching the impact of diet on dental health. Have you come across any studies l |
| 0.002 | oasst | pseudo | What other vegan recipes do you have? |
| 0.002 | oasst | pseudo | What about if you're a vegetarian? |
| 0.003 | oasst | pseudo | Do you have any suggestion on a budget but still good guitar for learning purposes? |
| 0.003 | oasst | pseudo | You told me much about how I can make my children happy. Now what about my own happiness. Can you suggest a few actions I can do to be happier and less stressed? |
| 0.003 | oasst | pseudo | What would be the most healthy meal that I could order off of the McDonald's menu? |
| 0.003 | persona_turn | audited | The Battlefield Strategy Reenactment sounds perfect! It combines football positioning with historical learning. I'm excited to try it out. On another note, our school is considering adding more field trips next year. Any |
| 0.003 | oasst | pseudo | Tell me more about Tikal in Guatemala. I would like to know if the site is wheelchair accessible ? |
| 0.003 | oasst | pseudo | That's a lot of information to take in. I'm looking for a place that has a wide variety of food and is easy to move around in. Can you recommend a place that meets this criteria? |
| 0.003 | oasst | pseudo | How would I begin training for a marathon as a man who is pretty overweight for his size and has no previous marathon or running experience? Give me a step by step process and some tips, along with encouragement at the e |
| 0.004 | oasst | pseudo | whats a good bread recipe that uses gluten free ingredients and takes about an hour to cook? |

