"""Connect array namespace to NN namespace."""

__all__: tuple[str, ...] = ()

from typing import Protocol, cast, overload

from stream_mapper.core.typing import Array, ArrayNamespace

# ===============================================================


class XPNamespaceMap(Protocol):
    """Protocol for mapping array namespaces to XP namespaces."""

    @overload
    def __getitem__(self, key: None) -> None: ...

    @overload
    def __getitem__(
        self, key: ArrayNamespace[Array] | str
    ) -> ArrayNamespace[Array]: ...

    def __getitem__(
        self, key: ArrayNamespace[Array] | str | None
    ) -> ArrayNamespace[Array] | None:
        """Get item."""
        ...

    # ---------------------------------------------------------------

    @overload
    def __setitem__(self, key: None, value: None) -> None: ...

    @overload
    def __setitem__(
        self, key: ArrayNamespace[Array] | str, value: ArrayNamespace[Array]
    ) -> None: ...

    def __setitem__(
        self,
        key: ArrayNamespace[Array] | str | None,
        value: ArrayNamespace[Array] | None,
    ) -> None:
        """Set item."""
        ...


XP_NAMESPACE = cast(XPNamespaceMap, {})
XP_NAMESPACE[None] = None

# ===============================================================


class XPReverseNamespaceMap(Protocol):
    """Protocol for mapping array namespaces to XP namespaces."""

    @overload
    def __getitem__(self, key: None) -> None: ...

    @overload
    def __getitem__(self, key: ArrayNamespace[Array]) -> str: ...

    def __getitem__(self, key: ArrayNamespace[Array] | None) -> str | None:
        """Get item."""
        ...

    # ---------------------------------------------------------------

    @overload
    def __setitem__(self, key: None, value: None) -> None: ...

    @overload
    def __setitem__(self, key: ArrayNamespace[Array], value: str) -> None: ...

    def __setitem__(self, key: ArrayNamespace[Array] | None, value: str | None) -> None:
        """Set item."""
        ...

    # ---------------------------------------------------------------

    def __contains__(self, key: ArrayNamespace[Array] | None) -> bool:
        """Check if key is in the map."""
        ...


XP_NAMESPACE_REVERSE = cast(XPReverseNamespaceMap, {})
XP_NAMESPACE_REVERSE[None] = None
