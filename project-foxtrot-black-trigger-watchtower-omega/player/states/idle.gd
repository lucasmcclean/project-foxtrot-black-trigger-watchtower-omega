extends PlayerState

@export var fall_state: State
@export var jump_state: State
@export var move_state: State



func enter() -> void:
	super()
	player.velocity.x = 0
	if not player.animation.is_playing():
		player.animation.play("idle")


func update(_delta: float):
	if not player.animation.is_playing():
		player.animation.play("idle")


func physics_update(delta: float) -> void:
	if player.input.jump and player.is_grounded():
		print("Jumping")
		state_machine.change_state(jump_state)
	
	player.velocity.y += player.gravity*delta
	player.move_and_slide()
	
	if !player.is_grounded():
		state_machine.change_state(fall_state)
	elif player.input.move.x != 0:
		state_machine.change_state(move_state)
