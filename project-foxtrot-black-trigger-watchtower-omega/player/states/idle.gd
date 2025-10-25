extends PlayerState

@export var fall_state: State
@export var jump_state: State
@export var move_state: State



func enter() -> void:
	player.get_node("Sprite2D").modulate = Color(0.981, 0.654, 0.284, 1.0) # Light blue tint for falling
	super()
	player.velocity.x = 0


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
