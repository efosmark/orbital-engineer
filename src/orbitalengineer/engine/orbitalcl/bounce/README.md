# Bounce

A kernel that will cause colliding bodies to bounce off of each-other.

## Coefficient of Restitution

- The coefficient of restitution (denoted as $e$ below), is a value between `0` and `1` that indicates how elastic the collision will be.
- A value of `1` will make the collision fully repulsive (no energy loss), whereas a value of `0` will be attenuated (full energy loss) and "stick" together.
- More information: [Coefficient of Restituion](https://en.wikipedia.org/wiki/Coefficient_of_restitution)
- The default value is defined in the config via the field `COEF_OF_RESTITUTION`.

## Algorithm

These steps are ran for each colliding pair $(i, j)$.

1. Normalize the difference in position

   $$
   r_{norm} = |r_j - r_i|
   $$

2. Find the relative speed along normal

   $$
   v_{nrel} = \operatorname{Re}((v_i - v_j) \cdot \overline{r_{norm}})
   $$

3. Get the scalar impulse magnitude

   $$
   \frac{1.0 + (e \cdot v_{nrel})}{
      \frac{1}{M_i} + \frac{1}{M_j}
   }
   $$

4. Apply the final impulse

   $$
    V_i = V_i + \frac{r_{norm} \cdot \text{scalar impulse}}{M_i}
   $$
