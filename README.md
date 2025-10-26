# Crimson Brawl

Crimson Brawl is a two player EEG-powered game. Instead of using a keyboard or
controller, players uses their heads to control game actions making gaming more
immersive, accessible, and interactive.

> Submitted on
> [Devpost](https://devpost.com/software/project-foxtrot-black-trigger-watchtower-omega/edit)
> for Knight Hacks VIII

## Inspiration

We wanted to explore how EEG technology could make gaming more immersive and
accessible for players who have physical disabilities. Relying only on the
player's head in order to move the character rather than just keyboard seemed
like it would be not only practical, but also fun.

## What it does

Our two player EEG powered game lets players use their head in order to control
in game actions and movements such as jumping, moving left and right, kicking,
punching, and dashing. All the controls are available just from jaw and head
movements.

## How we built it

To process the EEG data, we employed a decision tree based XGBoost model to
classify the signals as commands. For the game component we used the Godot game
engine. The player component is abstracted over a keyboard and EEG controller
allowing flexibility in how its played. Two connect the two, we stream the EEG
classified data as JSON over a WebSocket and process it in the Godot EEG player
controller.

## Challenges we ran into

- Implementing sprite animations cleanly into Godot
- Fluid player controls
- Designing intuitive and accurate EEG controls
- Balancing the keyboard player's level of control against the EEG player's

## Accomplishments that we're proud of

Successfully created a playable demo where two people can compete against one
another regardless of physical disability with character art and animation fully
completed within a day. We were exceptionally proud of the accuracy and
usability of the EEG as a player controller.

## What we learned

We gained a greater appreciation for the difficulty in designing games that are
fun, accessible, and balanced across input methods. We also gained a better
understanding of how to handle different input methods and handle input from the
network.

## What's next for Crimson Brawl

We plan to improve signal accuracy, add more levels, explore multiplayer. We'd
like to explore designing a more in-depth combat system that's still fully
accessible and balanced as well.
