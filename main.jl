using GLMakie
using FFTW

function avg_integral(zz, nrs)
    n1, n2 = size(zz)
    zz2 = zeros(3 * n1, 3 * n2)
    zz2[n1+1:2*n1, n2+1:2*n2] = zz
    zz3 = zeros(3 * n1, 3 * n2)
    kernel = zeros(3 * n1, 3 * n2)
    fftz = fft(zz2)

    for nr in nrs
        ksum = 0.0
        for i = -nr:nr
            ii = mod(i, 3 * n1) + 1
            for j = -nr:nr
                jj = mod(j, 3 * n2) + 1
                kernel[ii, jj] = (i^2 + j^2 < nr^2)
                ksum += kernel[ii, jj]
            end
        end
        for i = -nr:nr
            ii = mod(i, 3 * n1) + 1
            for j = -nr:nr
                jj = mod(j, 3 * n2) + 1
                kernel[ii, jj] /= ksum
            end
        end
        zz3 = max.(zz3, real.(ifft(fftz .* fft(kernel))))
    end
    return zz3
end

function hardy_llw(x, y, z, rmin, rmax)
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    n1, n2 = size(z)

    nrmin = Int(max(round(rmin / dx), 1))
    nrmax = Int(min(round(rmax / dx), n1 ÷ 2, n2 ÷ 2))

    T = range(0, 1.0, length = 21)
    nrs = [Int(round(nrmin * (1 .- t) + nrmax * t)) for t in T]
    nrs = unique(nrs)
    return avg_integral(z, nrs)
end

fig = Figure(resolution = (3840, 2160))

ax1 = Axis3(
    fig[1, 2],
    title = "3D plot",
    titlegap = 48, titlesize = 36
)

ax2 = fig[1, 3] = Axis(fig,
    # borders
    aspect = 1, targetlimits = BBox(-1, 1, -1, 1),
    # title 
    title = "λ-levelset",
    titlegap = 48, titlesize = 36,
    # x-axis
    xautolimitmargin = (0, 0), xgridwidth = 2, xticklabelsize = 24,
    xticks = LinearTicks(11), xticksize = 18,
    # y-axis
    yautolimitmargin = (0, 0), ygridwidth = 2, yticklabelpad = 14,
    yticklabelsize = 24, yticks = LinearTicks(11), yticksize = 18
)



r_range = IntervalSlider(
    fig[2, 2],
    range = range(0, 1, length = 101),
    startvalues = (0.0, 0.5),
    label = "lol",
    format = [x -> "$(round(x, digits = 1))" for i = 1:2],
)

rmin = @lift(($(r_range.interval))[1])
rmax = @lift(($(r_range.interval))[2])
λ = Slider(fig[3, 2], range = range(0, 10, length = 10001), startvalue = 3).value
fparam = Slider(fig[4, 2], range = range(0, 1.8, length = 101), startvalue = 1).value
fparam2 = Slider(fig[5, 2], range = range(0, 1, length = 101), startvalue = 0.5).value

label_rmin_max = lift(r_range.interval) do r
    string(round.(r, digits = 2))
end
Label(fig[2, 1], label_rmin_max)
Label(fig[3, 1], @lift("λ = " * string(round.($λ, digits = 2))))
Label(fig[4, 1], @lift("f_exp = " * string(round.($fparam, digits = 2))))
Label(fig[5, 1], @lift("f_den = " * string(round.($fparam2, digits = 2))))

x = range(-1, 1, length = 200)
y = range(-1, 1, length = 200)
z = @lift(@. abs.(3 - 1 / sqrt((x')^2 + y^2 + $fparam2)^$fparam))  # broadcasts to 2d array
zlims!(ax1, (0, 5)) # as tuple, reversed

dx = x[2] - x[1]
f_l1_norm = @lift(dx * dx * sum(($z)))
Label(fig[2, 3], @lift("||f||_1 = " * string(round.($f_l1_norm, digits = 3))), tellwidth = false)

levs = @lift([-100, $λ, 1000000])
surface!(ax1, x, y, z, colormap = :deep, transparency = true)



llw = @lift(hardy_llw(x, y, $z, $rmin, $rmax))
llw_plot = @lift($llw[201:400, 201:400])
surface!(ax1, x, y, llw_plot, transparency = true)


contour!(ax2, x, y, llw_plot, levels = levs)
contourf!(ax2, x, y, llw_plot, levels = levs)


llset_size = @lift(dx * dx * sum(($llw) .> $λ))
Label(fig[3, 3], @lift("μ(a_λ) = " * string(round.($llset_size, digits = 3))), tellwidth = false)
Label(fig[4, 3], @lift("C ≥ " * string(round.(($llset_size) * ($λ) / ($f_l1_norm), digits = 3))), tellwidth = false)




klm = 0