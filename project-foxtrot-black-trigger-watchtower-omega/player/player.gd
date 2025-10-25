class_name Player
extends CharacterBody2D

var gravity: float = 8000.0
var jump_gravity: float = 800.0
var acceleration: float = 10000.0
var friction: float = 25000.0
var air_acceleration: float = 9900.0
var air_friction: float = 24000.0
var h_speed: float = 300.0
var v_speed: float = 3000.0
var jump_impulse: float = 1500.0

@export var input: PlayerInput
@onready var controller: Controller = $Controller
@onready var state_machine: StateMachine = $StateMachine

var health: int = 20


func _ready() -> void:
	state_machine.initialize()


func _unhandled_input(event: InputEvent) -> void:
	state_machine.handle_input(event)


func _physics_process(delta: float) -> void:
	controller.sample_input()
	state_machine.physics_update(delta)


func _process(delta: float) -> void:
	state_machine.update(delta)


#TODO find most efficient way to ground check
func is_grounded() -> bool:
	for cast in $GroundCasts.get_children():
		if cast.is_colliding():
			return true
	return false
