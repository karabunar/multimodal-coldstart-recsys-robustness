## other designs of noise-augmented training at rate 0.4 (hashtag test noise 0.4)

| design | seed | best epoch | noisy Recall@20 | clean-trained, same seed | clean-trained + M3 | clean Recall@20 | w_t on clean cold items | w_t on warm items |
|---|---|---|---|---|---|---|---|---|
| A shuffle | 2025 | 390 | 0.0493 | 0.0667 | 0.0780 | 0.0636 | 0.988 | 0.418 |
| A shuffle | 2026 | 140 | 0.0578 | 0.0652 | 0.0731 | 0.0642 | 0.986 | 0.406 |
| A shuffle | 2027 | 280 | 0.0559 | 0.0685 | 0.0773 | 0.0662 | 0.968 | 0.423 |
| B fixed prototypes | 2025 | 530 | 0.0683 | 0.0667 | 0.0780 | 0.0777 | 0.941 | 0.478 |
| B fixed prototypes | 2026 | 350 | 0.0748 | 0.0652 | 0.0731 | 0.0732 | 0.946 | 0.466 |
| B fixed prototypes | 2027 | 450 | 0.0698 | 0.0685 | 0.0773 | 0.0732 | 0.962 | 0.465 |
| C image side | 2025 | 520 | 0.0616 | 0.0667 | 0.0780 | 0.0760 | 0.323 | 0.384 |
| C image side | 2026 | 680 | 0.0589 | 0.0652 | 0.0731 | 0.0695 | 0.336 | 0.387 |
| C image side | 2027 | 440 | 0.0646 | 0.0685 | 0.0773 | 0.0744 | 0.301 | 0.393 |
