# Predicting MLB Performance

This project is my attempt to predict the winner of a given MLB game. All data is collected from pybaseball and webscraping sites such as MLB.com, Fangraphs, baseball-reference, and sportsbookreview (for betting odds). While the main goal is if we can develop a solid model to predict team performance, the moneyline odds and betting allows us to see not only how often the model may be correct, but how confident it is in the picks that it suggests.

With baseball being a game surrounded by statistics, I wanted to attempt to simplify by creating categories for pitchers and a single numeric metric for bullpen rankings. The pitcher categories allow us to look at how a team performed against similar pitcher in the same category, as looking at team-pitcher comparisons proves useful but a team may only face the same pitcher a couple times a season. 

To see how the model performs over a prolonged period of time, we incorporate a starting account balance and have the model bet on the picks it is most confident in. The obvious end goal is to have a model that can surpass it's starting account balance by the end of a season or longer.
