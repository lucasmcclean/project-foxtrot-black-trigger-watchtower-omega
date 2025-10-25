class_name Player
extends CharacterBody2D

var gravity: float = 8000.0
var jump_gravity: float = 800.0
var acceleration: float = 10000.0
var friction: float = 25000.0
var air_acceleration: float = 9900.0
var air_friction: float = 24000.0
var h_min_speed: float = 300.0
var h_speed: float = 300.0
var v_speed: float = 3000.0
var jump_impulse: float = 1500.0


@export var input: PlayerInput
@onready var controller: Controller = $Controller
@onready var state_machine: StateMachine = $StateMachine
@export var  move_state: State

@onready var punch_hitbox: Area2D = $PunchHitbox
@onready var kick_hitbox: Area2D = $KickHitbox

@onready var hurtbox: Area2D = $Hurtbox


var health: int = 20

var flash_distance: float = 500.0
var flash_impulse: float = 10000.0
var flash_cooldown: float = 3.0
var can_flash: bool = true 

func _ready() -> void:
	state_machine.initialize()


func _unhandled_input(event: InputEvent) -> void:
	state_machine.handle_input(event)


func _physics_process(delta: float) -> void:
	controller.sample_input()
	state_machine.physics_update(delta)
	if input.punch:
		perform_attack(punch_hitbox)
		
	if input.kick:
		perform_attack(kick_hitbox)
	
	if input.flash_step and can_flash and is_grounded():
		hurtbox.monitorable = false
		hurtbox.monitoring = false
		handle_flash_step()
	else:
		hurtbox.monitorable = true
		hurtbox.monitoring = true


func _process(delta: float) -> void:
	state_machine.update(delta)


#TODO find most efficient way to ground check
func is_grounded() -> bool:
	for cast in $GroundCasts.get_children():
		if cast.is_colliding():
			return true
	return false


func perform_attack(hitbox: Area2D) -> void:
	var overlapping_areas = hitbox.get_overlapping_areas()
	for area in overlapping_areas:
		if area.name == "Hurtbox" and area.get_parent() != self:
			print("taking damage")
			area.get_parent().health -= 1


func _on_hurtbox_area_entered(area: Area2D) -> void:
	if not area is Hitbox:
		return

	health -= 1
	
func handle_flash_step() -> void:
	h_speed = 30000.0
	can_flash = false
	velocity.x += flash_impulse
	state_machine.change_state(move_state)
	await get_tree().create_timer(flash_cooldown).timeout
	can_flash = true
