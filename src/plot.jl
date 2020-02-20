import Statistics
import PyPlot
import Plots

function positions(tree)

        # convert tree to tabular form, and add position columns
        tree_tab = tree_to_array(tree)
        tree_tab_xy = hcat((tree_tab),zeros(size(tree_tab)[1],2))

        depth = maximum(tree_tab[:,7]) + 1
        nleaf = sum(tree_tab[:,3])
        y_depths = collect(1:-1/(depth-1):0)

        # position of leaf nodes
        tree_tab_xy[tree_tab[:,3] .== true,8] = collect(0:1/(nleaf-1):1)
        tree_tab_xy[tree_tab[:,3] .== true,9] .= 0.0

        for i in depth:-1:1
                nodes_at_depth_i = tree_tab_xy[tree_tab_xy[:,7] .== (i-1),:]
                for j in 1:size(nodes_at_depth_i)[1]
                        if nodes_at_depth_i[j,3] == false
                                xpos = Statistics.mean(tree_tab_xy[tree_tab_xy[:,2] .== nodes_at_depth_i[j,1],8])
                                ypos = y_depths[i]
                                tree_tab_xy[tree_tab_xy[:,1] .== nodes_at_depth_i[j,1],8] .= xpos
                                tree_tab_xy[tree_tab_xy[:,1] .== nodes_at_depth_i[j,1],9] .= ypos
                        end
                end
        end
        return(tree_tab_xy)
end

function lines(tree_tab_xy)
        xpos = []
        ypos = []
        for i in 1:size(tree_tab_xy)[1]
                # check not root
                if (tree_tab_xy[i,2] != -1)
                        x = reshape(tree_tab_xy[i,[8,9]],(2))
                        push!(xpos,x)
                        y = reshape(tree_tab_xy[tree_tab_xy[:,1] .== tree_tab_xy[i,2],[8,9]],(2))
                        push!(ypos,y)
                end
        end
        return(xpos,ypos)
end

function add_plot_text(tree_tab_xy)
        txt = repeat(["."],size(tree_tab_xy)[1])
        txt[tree_tab_xy[:,3] .== true] .= string.(round.(tree_tab_xy[tree_tab_xy[:,3] .== true,4],digits=1))
        txt[tree_tab_xy[:,3] .== false] .= "X" .* string.(tree_tab_xy[tree_tab_xy[:,3] .== false,5]) .* " < " .*
                        string.(round.(tree_tab_xy[tree_tab_xy[:,3] .== false,6],digits=1)) .* "\nsamples = n"
        txt = reshape(txt,(length(txt),1))
        tree_tab_xy = hcat((tree_tab_xy),txt)
        return(tree_tab_xy)
end

"""
Helper function
"""
function rectangle(w, h, x, y)
        Plots.Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
end



function plot(tree;split_xsize=1,split_ysize=1,leaf_size=1)
        Plots.pyplot()
        points_xy = positions(tree.tree)
        lines_xy = lines(points_xy)
        points_xy = add_plot_text(points_xy)

        mshape = repeat([:circle],sum(points_xy[:,3]))
        mcolor = repeat([Plots.RGB(230/255,230/255,250/255)],size(points_xy)[1])
        mcolor[points_xy[:,3] .== true] .= Plots.RGB(245/255,222/255,179/255)

        p = Plots.plot()
        # plot lines
        for i in 1:(size(points_xy)[1]-1)
                Plots.plot!(p,[lines_xy[1][i][1],lines_xy[2][i][1]],[lines_xy[1][i][2],lines_xy[2][i][2]],
                linecolor=:black,
                linewidth=3)
        end

        for i in 1:size(points_xy)[1]
                if points_xy[i,3] == false
                        Plots.plot!(p,
                                rectangle(0.1,0.1,points_xy[i,8]-0.1/2,points_xy[i,9]-0.1/2),
                                colour=Plots.RGB(230/255,230/255,250/255),
                                annotations=(points_xy[i,8],points_xy[i,9],
                                Plots.text.(points_xy[i,10],10)))
                end
        end

        leafm = convert(Array{Union{Bool},1},points_xy[:,3] .== true)
        Plots.scatter!(p,points_xy[leafm,8],points_xy[leafm,9],
                legend=false,axis=false,
                  markershape=mshape,
                  grid = false,
                  markersize=8*leaf_size,
                  xlims=(-0.05,1.05),
                  ylims=(-0.06,1.06),
                  markercolor=mcolor[leafm],
                  markerstrokealpha=0,
                  annotations=(points_xy[leafm,8],points_xy[leafm,9],
                  Plots.text.(points_xy[leafm,10],10)))

        Plots.annotate!(p,[(points_xy[1,8]-0.16,points_xy[1,9]-0.1,"True")])
        Plots.annotate!(p,[(points_xy[1,8]+0.16,points_xy[1,9]-0.1,"False")])
        display(p)
end
