class_name EEGInputController
extends Controller

@export var websocket_url: String = "ws://0.0.0.0:8765"

signal connected
signal disconnected
signal message_received(msg: String)

var socket := WebSocketPeer.new()
var latest_data: Dictionary = {}
var _was_connected := false

func _ready():
	var err = socket.connect_to_url(websocket_url)
	if err != OK:
		push_error("Unable to connect to EEG server.")
		set_process(false)
	else:
		print("Connecting to EEG server: %s" % websocket_url)

func _process(_delta):
	socket.poll()
	_update_connection_state()
	_process_packets()

func _update_connection_state():
	var state = socket.get_ready_state()
	if state == WebSocketPeer.STATE_OPEN:
		if not _was_connected:
			_was_connected = true
			print("EEG WebSocket connected")
			emit_signal("connected")
	elif state in [WebSocketPeer.STATE_CLOSING, WebSocketPeer.STATE_CLOSED]:
		if _was_connected:
			_was_connected = false
			print("EEG WebSocket disconnected")
			emit_signal("disconnected")

func _process_packets():
	while socket.get_available_packet_count() > 0:
		var packet = socket.get_packet()
		if socket.was_string_packet():
			var text = packet.get_string_from_utf8()
			emit_signal("message_received", text)
			_parse_message(text)

func _parse_message(message: String):
	var data = JSON.parse_string(message)
	if data == null:
		push_error("Failed to parse EEG message.")
		return

	if "move" in data:
		latest_data["move"] = Vector2(data["move"][0], data["move"][1])

	for key in ["jump", "punch", "kick", "flash_step"]:
		if key in data:
			latest_data[key] = data[key]

func sample_input():
	if player.input == null:
		return

	if "move" in latest_data:
		player.input.move = latest_data["move"]

	for key in ["jump", "punch", "kick", "flash_step"]:
		if key in latest_data:
			player.input[key] = latest_data[key]
