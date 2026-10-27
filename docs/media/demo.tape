# Regenerate the README demo from the repo root:  vhs docs/media/demo.tape
Output docs/media/demo.gif

Require python3
Require pygmentize

Set Shell "bash"
Set Theme "GitHub Dark"
Set FontSize 15
Set Width 1240
Set Height 560
Set Padding 22
Set WindowBar Colorful
Set BorderRadius 10
Set TypingSpeed 35ms

Type "sed -n 8,31p cuda/06-gemm-tiled.cu | pygmentize -l cuda -f terminal256 -O style=github-dark"
Sleep 300ms
Enter
Sleep 5s

Type "clear"
Enter
Sleep 300ms

Type "for f in reference/*.py; do python3 $f; done"
Sleep 300ms
Enter
Sleep 5s
