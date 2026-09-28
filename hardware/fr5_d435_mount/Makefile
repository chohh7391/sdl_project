# Re-export the printable bodies from the one parametric source.
#   sudo apt install openscad      (or point OPENSCAD= at an AppImage)
OPENSCAD ?= openscad
# Two mounts, one per imager: the cradle is offset so that imager, not the
# camera body, lands on the tool axis. Print the one your detector runs on.
all: mount_color.stl mount_ir.stl gauge_flange.stl gauge_camera.stl

mount_color.stl: fr5_d435_mount.scad
	$(OPENSCAD) -D 'part="mount"' -D 'cam_lens_dy=32.5' -o $@ $<

mount_ir.stl: fr5_d435_mount.scad
	$(OPENSCAD) -D 'part="mount"' -D 'cam_lens_dy=17.5' -o $@ $<

gauge_%.stl: fr5_d435_mount.scad
	$(OPENSCAD) -D 'part="gauge_$*"' -o $@ $<

check: mount_color.stl mount_ir.stl
	python3 check_geometry.py mount_color.stl 32.5
	python3 check_geometry.py mount_ir.stl 17.5

preview: mount_color.stl
	python3 preview.py

clean:
	rm -f mount_color.stl mount_ir.stl gauge_flange.stl gauge_camera.stl

.PHONY: all check preview clean
