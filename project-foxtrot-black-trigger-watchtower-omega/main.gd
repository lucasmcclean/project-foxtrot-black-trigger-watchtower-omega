extends Node2D

@onready var local_player: Player = $LocalPlayer
@onready var eeg_player: Player = $EEGPlayer

func _physics_process(_delta: float) -> void:
	if local_player.health <= 0:
		print("EEG player wins")
		reset_game()
	elif  eeg_player.health <= 0:
		print("Local player wins")
		reset_game()

func reset_game() -> void:
	local_player.position = Vector2(1100, 560)
	eeg_player.position = Vector2(50, 560)
