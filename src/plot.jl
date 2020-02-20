"""
Algorithm for calculating the (x,y) coordinates for
        positions of nodes in decision trees
"""
function positions(tree::Node)

        # convert tree to tabular form, and add position columns
        tree_tab = tree_to_array(tree)
        tree_tab_xy = hcat((tree_tab),zeros(size(tree_tab)[1],2))

        depth = maximum(tree_tab[:,7]) + 1
        nleaf = sum(tree_tab[:,3])
        y_depths = collect(1:-1/(depth-1):0)

        # position of leaf nodes
        tree_tab_xy[tree_tab[:,3] .== true,9] = collect(0:1/(nleaf-1):1)
        tree_tab_xy[tree_tab[:,3] .== true,10] .= 0.0

        for i in depth:-1:1
                nodes_at_depth_i = tree_tab_xy[tree_tab_xy[:,7] .== (i-1),:]
                for j in 1:size(nodes_at_depth_i)[1]
                        if nodes_at_depth_i[j,3] == false
                                xpos = Statistics.mean(tree_tab_xy[tree_tab_xy[:,2] .== nodes_at_depth_i[j,1],9])
                                ypos = y_depths[i]
                                tree_tab_xy[tree_tab_xy[:,1] .== nodes_at_depth_i[j,1],9] .= xpos
                                tree_tab_xy[tree_tab_xy[:,1] .== nodes_at_depth_i[j,1],10] .= ypos
                        end
                end
        end
        return(tree_tab_xy)
end

"""
Algorithm for
"""
function lines(tree_tab_xy)
        xpos = []
        ypos = []
        for i in 1:size(tree_tab_xy)[1]
                # check not root
                if (tree_tab_xy[i,2] != -1)
                        x = reshape(tree_tab_xy[i,[9,10]],(2))
                        push!(xpos,x)
                        y = reshape(tree_tab_xy[tree_tab_xy[:,1] .== tree_tab_xy[i,2],[9,10]],(2))
                        push!(ypos,y)
                end
        end
        return(xpos,ypos)
end

"""
Add
"""
function add_plot_text(tree_tab_xy)
        txt = repeat(["."],size(tree_tab_xy)[1])
        mskl = convert(Array{Union{Bool},1},tree_tab_xy[:,3] .== true)
        txt[mskl] .= string.(round.(tree_tab_xy[mskl,4],digits=1))
        mskn = convert(Array{Union{Bool},1},tree_tab_xy[:,3] .== false)
        txt[mskn] .= "X" .* string.(tree_tab_xy[mskn,5]) .* " < " .*
                        string.(round.(tree_tab_xy[mskn,6],digits=1)) .*
                        "\nn = " .* string.(tree_tab_xy[mskn,8])
        txt = reshape(txt,(length(txt),1))
        tree_tab_xy = hcat((tree_tab_xy),txt)
        return(tree_tab_xy)
end

"""
Plot a rectangle
"""
function rectangle(w, h, x, y)
        # b = bottom, t = top, l = left, r = right
        bl = (x-w/2,y-h/2)
        br = (x+w/2,y-h/2)
        tr = (x+w/2,y+h/2)
        tl = (x-w/2,y+h/2)
        Plots.Shape([bl,br,tr,tl,bl])
end

"""
Plot a tree
"""
function plot(tree::Tree;split_xsize=1,split_ysize=1,leaf_size=1)
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


        xsize = split_xsize*0.1
        ysize = split_ysize*0.1
        for i in 1:size(points_xy)[1]
                if points_xy[i,3] == false
                        Plots.plot!(p,
                                rectangle(xsize,ysize,
                                points_xy[i,9],points_xy[i,10]),
                                markerstrokealpha = 0.2,
                                linealpha = 1.0,
                                colour=Plots.RGB(230/255,230/255,250/255),
                                annotations=(points_xy[i,9],points_xy[i,10],
                                Plots.text.(points_xy[i,11],10)))
                end
        end

        leafm = convert(Array{Union{Bool},1},points_xy[:,3] .== true)
        Plots.scatter!(p,points_xy[leafm,9],points_xy[leafm,10],
                legend=false,axis=false,
                  markershape=mshape,
                  grid = false,
                  markersize=8*leaf_size,
                  xlims=(-0.05,1.05),
                  ylims=(-0.06,1.06),
                  markercolor=mcolor[leafm],
                  markerstrokealpha=0,
                  annotations=(points_xy[leafm,9],points_xy[leafm,10],
                  Plots.text.(points_xy[leafm,11],10)))

        Plots.annotate!(p,[(points_xy[1,9]-0.16,points_xy[1,10]-0.1,"True")])
        Plots.annotate!(p,[(points_xy[1,9]+0.16,points_xy[1,10]-0.1,"False")])
        display(p)
end
