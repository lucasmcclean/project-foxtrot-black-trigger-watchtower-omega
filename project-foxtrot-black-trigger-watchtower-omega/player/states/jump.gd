extends PlayerState

@export var fall_state: State
@export var idle_state: State
@export var move_state: State

var just_jumped: bool = false


func enter() -> void:
	player.get_node("Sprite2D").modulate = Color(0.0, 0.848, 0.779, 1.0) # Light blue tint for falling

	if(!just_jumped):
		super()
		just_jumped = true
		player.velocity.y -= player.jump_impulse
		just_jumped_timer()
		


func physics_update(delta: float) -> void:
	player.velocity.y += player.jump_gravity*delta
	if (player.input.move.x != 0) and (player.velocity.x*player.input.move.x > 0):
		player.velocity.x += player.input.move.x*player.air_acceleration*delta
	elif player.input.move.x != 0:
		player.velocity.x += player.input.move.x*player.air_friction*delta
	else:
		player.velocity.x = move_toward(player.velocity.x, 0, player.air_friction*delta)
	player.velocity.x = clamp(player.velocity.x, -player.h_speed, player.h_speed)
	player.velocity.y = clamp(player.velocity.y, -player.v_speed, player.v_speed)
	player.move_and_slide()
	
	if player.is_grounded() and !is_zero_approx(player.velocity.x) and !just_jumped:
		print(player.is_grounded())
		state_machine.change_state(move_state)
	elif player.is_grounded() and !just_jumped:
		state_machine.change_state(idle_state)
	elif player.input.move.y >= 0:
		state_machine.change_state(fall_state)

#TODO make sure this works and isn't buggy
func just_jumped_timer() -> void:
	await get_tree().create_timer(0.5).timeout
	just_jumped = false
